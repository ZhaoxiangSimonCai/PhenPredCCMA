import os
import shap
import torch
import pickle
import PhenPred
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary
from datetime import datetime
from PhenPred.vae import plot_folder, shap_folder
from torch.utils.data import DataLoader
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Model import (
    MOSA,
    DiffusionMOSA,
    DiffusionScheduler,
    TransformerMOSA,
    TransformerDiffusionMOSA,
)
from PhenPred.vae.ModelGMVAE import GMVAE
from PhenPred.vae.Losses import CLinesLosses
import h5py
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)


class CLinesTrain:
    def __init__(
        self,
        data,
        hypers,
        stratify_cv_by=None,
        early_stop_patience=100,
        timestamp=None,
        verbose=0,
    ):
        self.data = data
        self.losses = []
        self.hypers = hypers
        self.stratify_cv_by = stratify_cv_by
        self.verbose = verbose

        self.timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else timestamp
        )

        self.early_stop_patience = early_stop_patience

        self.lrs = [(1, hypers["learning_rate"])]
        self.benchmark_scores = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transfer_loaded_views = []
        self.transfer_freeze_epochs = int(
            self.hypers.get("transfer_freeze_epochs", 0) or 0
        )
        self._transfer_unfrozen = self.transfer_freeze_epochs <= 0
        self._cached_transfer_state = None
        self._cached_transfer_source_views = None

    def run(self, run_timestamp=None, return_val_loss=False):
        if run_timestamp is not None:
            self.timestamp = run_timestamp
            return

        if not self.hypers["skip_cv"]:
            self.training(drop_last=True, skip_cv_save=return_val_loss)
            if return_val_loss:
                return pd.DataFrame(self.losses)
            losses_df = self.save_losses()
            self.plot_losses(losses_df)

        self.predictions(drop_last=True)

        if self.hypers["save_model"]:
            self.save_model()

    @staticmethod
    def _strip_module_prefix(state_dict):
        return {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }

    @staticmethod
    def _map_view_index_key(key, source_view_names, target_view_names):
        for prefix in ("encoders.", "decoders."):
            if key.startswith(prefix):
                remainder = key[len(prefix) :]
                parts = remainder.split(".", 1)
                if len(parts) != 2 or not parts[0].isdigit():
                    return None

                source_idx = int(parts[0])
                if source_idx >= len(source_view_names):
                    return None

                view_name = source_view_names[source_idx]
                if view_name not in target_view_names:
                    return None

                target_idx = target_view_names.index(view_name)
                return f"{prefix}{target_idx}.{parts[1]}"

        return key

    @staticmethod
    def _resolve_optional_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        if os.path.isfile(path):
            return path
        candidate = os.path.join(plot_folder, "files", path)
        if os.path.isfile(candidate):
            return candidate
        return None

    @staticmethod
    def _get_base_model(model):
        model_core = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(model_core, "base_mosa"):
            return model_core.base_mosa
        if hasattr(model_core, "base_transformer"):
            return model_core.base_transformer
        return model_core

    def _freeze_loaded_views(self, model, shared_views):
        if self.transfer_freeze_epochs <= 0 or not shared_views:
            self._transfer_unfrozen = True
            return

        base_model = self._get_base_model(model)
        target_view_names = list(self.hypers["datasets"].keys())

        for view_name in shared_views:
            view_idx = target_view_names.index(view_name)
            for module_name in ("encoders", "decoders"):
                modules = getattr(base_model, module_name, None)
                if modules is None or view_idx >= len(modules):
                    continue
                for param in modules[view_idx].parameters():
                    param.requires_grad = False

        self._transfer_unfrozen = False
        print(
            f"Transfer learning: frozen shared view enc/dec layers for "
            f"{self.transfer_freeze_epochs} epochs."
        )

    def maybe_unfreeze_transfer_modules(self, model, epoch):
        if self._transfer_unfrozen or self.transfer_freeze_epochs <= 0:
            return
        if epoch <= self.transfer_freeze_epochs:
            return

        base_model = self._get_base_model(model)
        target_view_names = list(self.hypers["datasets"].keys())

        for view_name in self.transfer_loaded_views:
            view_idx = target_view_names.index(view_name)
            for module_name in ("encoders", "decoders"):
                modules = getattr(base_model, module_name, None)
                if modules is None or view_idx >= len(modules):
                    continue
                for param in modules[view_idx].parameters():
                    param.requires_grad = True

        self._transfer_unfrozen = True
        print("Transfer learning: unfroze shared view enc/dec layers.")

    def load_pretrained_shared_views(self, model):
        if self.hypers.get("transfer_mode") != "shared_views_partial":
            return

        checkpoint_path = self._resolve_optional_path(
            self.hypers.get("transfer_checkpoint")
        )
        if checkpoint_path is None:
            warnings.warn(
                "transfer_mode='shared_views_partial' set but transfer_checkpoint "
                "is missing or unresolved; skipping transfer initialization."
            )
            return

        transfer_hypers_path = self._resolve_optional_path(
            self.hypers.get("transfer_hypers_json")
        )
        if transfer_hypers_path is None:
            warnings.warn(
                "transfer_hypers_json is required for shared-view mapping; "
                "skipping transfer initialization."
            )
            return

        if self._cached_transfer_state is None:
            source_hypers = Hypers.read_hyperparameters(
                hypers_json=transfer_hypers_path,
                parse_torch_functions=False,
            )
            source_view_names = list(source_hypers["datasets"].keys())

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                source_state = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                source_state = checkpoint["model_state_dict"]
            else:
                source_state = checkpoint

            if not isinstance(source_state, dict):
                warnings.warn("Unsupported checkpoint format; skipping transfer.")
                return

            self._cached_transfer_state = self._strip_module_prefix(source_state)
            self._cached_transfer_source_views = source_view_names
        else:
            source_view_names = self._cached_transfer_source_views

        target_view_names = list(self.hypers["datasets"].keys())
        shared_views = [v for v in target_view_names if v in source_view_names]

        if not shared_views:
            warnings.warn("No shared views between source and target configs.")
            return

        source_state = self._cached_transfer_state
        target_state = model.module.state_dict()

        mapped_state = {}
        skipped = 0

        for source_key, source_value in source_state.items():
            mapped_key = self._map_view_index_key(
                source_key,
                source_view_names=source_view_names,
                target_view_names=target_view_names,
            )

            if mapped_key is None:
                skipped += 1
                continue

            if mapped_key not in target_state:
                skipped += 1
                continue

            if tuple(target_state[mapped_key].shape) != tuple(source_value.shape):
                skipped += 1
                continue

            mapped_state[mapped_key] = source_value

        model.module.load_state_dict(mapped_state, strict=False)
        self.transfer_loaded_views = shared_views
        print(
            "Transfer learning: loaded "
            f"{len(mapped_state)} tensors ({len(shared_views)} shared views: "
            f"{', '.join(shared_views)}), skipped {skipped} tensors."
        )

        self._freeze_loaded_views(model, shared_views)

    def initialize_model(self):
        views_sizes = {n: v.sum() for n, v in self.data.features_mask.items()}
        views_sizes_full = None

        if self.hypers["filtered_encoder_only"]:
            views_sizes_full = {n: v.shape[1] for n, v in self.data.views.items()}

        assert self.hypers["model"] in [
            "MOSA",
            "GMVAE",
            "DiffusionMOSA",
            "TransformerMOSA",
            "TransformerDiffusionMOSA",
        ], "Invalid model"

        if self.hypers["model"] == "MOSA":
            model = MOSA(
                hypers=self.hypers,
                views_sizes=views_sizes,
                conditional_size=(
                    self.data.labels.shape[1] if self.hypers["use_conditionals"] else 0
                ),
                views_sizes_full=views_sizes_full,
            )
        elif self.hypers["model"] == "DiffusionMOSA":
            base_mosa = MOSA(
                hypers=self.hypers,
                views_sizes=views_sizes,
                conditional_size=(
                    self.data.labels.shape[1] if self.hypers["use_conditionals"] else 0
                ),
                views_sizes_full=views_sizes_full,
            )
            model = DiffusionMOSA(
                base_mosa,
                DiffusionScheduler(
                    num_timesteps=self.hypers.get("diffusion_steps", 1000),
                    beta_start=self.hypers.get("beta_start", 1e-4),
                    beta_end=self.hypers.get("beta_end", 0.02),
                ),
            )
        elif self.hypers["model"] == "TransformerMOSA":
            # Replace MOSA with TransformerMOSA in your training code
            model = TransformerMOSA(
                hypers=self.hypers,
                views_sizes=views_sizes,
                conditional_size=(
                    self.data.labels.shape[1] if self.hypers["use_conditionals"] else 0
                ),
                views_sizes_full=views_sizes_full,
            )
        elif self.hypers["model"] == "TransformerDiffusionMOSA":
            # Enhanced TransformerMOSA with diffusion
            base_transformer = TransformerMOSA(
                hypers=self.hypers,
                views_sizes=views_sizes,
                conditional_size=(
                    self.data.labels.shape[1] if self.hypers["use_conditionals"] else 0
                ),
                views_sizes_full=views_sizes_full,
            )
            model = TransformerDiffusionMOSA(
                base_transformer,
                DiffusionScheduler(
                    num_timesteps=self.hypers.get("diffusion_steps", 1000),
                    beta_start=self.hypers.get("beta_start", 1e-4),
                    beta_end=self.hypers.get("beta_end", 0.02),
                ),
            )
        else:
            model = GMVAE(
                hypers=self.hypers,
                views_sizes=views_sizes,
                views_sizes_full=views_sizes_full,
                conditional_size=(
                    self.data.labels.shape[1] if self.hypers["use_conditionals"] else 0
                ),
            )

        model = nn.DataParallel(model)
        self.load_pretrained_shared_views(model)

        print(summary(model))

        return model

    def epoch(
        self,
        model,
        optimizer,
        dataloader,
        record_losses=None,
    ):
        for data in dataloader:
            x, y, x_nans, x_mask = data

            x = [m.to(self.device) for m in x]
            x_nans = [m.to(self.device) for m in x_nans]

            x_masked = [m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x)]
            x_nans_masked = [
                m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x_nans)
            ]

            y = y.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(model.training):
                if self.hypers["use_conditionals"]:
                    out_net = model(x_masked + [y])
                else:
                    out_net = model(x_masked)

                if self.hypers["model"] not in [
                    "DiffusionMOSA",
                    "TransformerDiffusionMOSA",
                ]:
                    loss = model.module.loss(
                        x if self.hypers["filtered_encoder_only"] else x_masked,
                        (
                            x_nans
                            if self.hypers["filtered_encoder_only"]
                            else x_nans_masked
                        ),
                        out_net,
                        y,
                        x_mask,
                        view_loss_weights=self.hypers["view_loss_weights"],
                    )
                elif self.hypers["model"] == "DiffusionMOSA":
                    vae_loss = model.module.base_mosa.loss(
                        x if self.hypers["filtered_encoder_only"] else x_masked,
                        (
                            x_nans
                            if self.hypers["filtered_encoder_only"]
                            else x_nans_masked
                        ),
                        out_net,
                        y,
                        x_mask,
                        view_loss_weights=self.hypers["view_loss_weights"],
                    )
                    loss = model.module.combined_loss(
                        out_net,
                        vae_loss,
                        diffusion_weight=self.hypers.get("diffusion_weight", 1.0),
                    )
                else:  # TransformerDiffusionMOSA
                    transformer_loss = model.module.base_transformer.loss(
                        x if self.hypers["filtered_encoder_only"] else x_masked,
                        (
                            x_nans
                            if self.hypers["filtered_encoder_only"]
                            else x_nans_masked
                        ),
                        out_net,
                        y,
                        x_mask,
                        view_loss_weights=self.hypers["view_loss_weights"],
                    )
                    loss = model.module.combined_loss(
                        out_net,
                        transformer_loss,
                        diffusion_weight=self.hypers.get("diffusion_weight", 1.0),
                    )

                if model.training:
                    total_loss = loss["total"]
                    if not torch.isfinite(total_loss):
                        warnings.warn(
                            "Non-finite training loss encountered; "
                            "skipping optimizer step for this batch."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    total_loss.backward()

                    grad_clip_max_norm = self.hypers.get("grad_clip_max_norm")
                    if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float(grad_clip_max_norm)
                        )

                    optimizer.step()

            if record_losses is not None:
                self.register_loss(loss, record_losses)

                if self.verbose > 1:
                    self.benchmarks(x, y, out_net["x_hat"], record_losses)

            else:
                self.print_single_loss(loss)

    def benchmarks(self, x, labels, x_hat, record_losses):
        # CDKN2A proteomics benchmark
        f = "CDKN2A"

        prot_idx = self.data.get_view_feature_index(f, "proteomics")
        prot_view_index = self.data.view_names.index("proteomics")
        prot_pred = x_hat[prot_view_index][:, prot_idx]

        if "copynumber" in self.data.view_names:
            cnvs_idx = self.data.get_view_feature_index(f, "copynumber")
            cnvs_view_index = self.data.view_names.index("copynumber")
            cnvs_true = x[cnvs_view_index][:, cnvs_idx]
        else:
            cnvs_idx = self.data.labels_name.index(f"cnv_{f}")
            cnvs_true = labels[:, cnvs_idx]

        # check if there are any CNVs with -2 value
        f_score = np.nanmedian(
            prot_pred[cnvs_true != -2].detach().numpy()
        ) - np.nanmedian(prot_pred[cnvs_true == -2].detach().numpy())

        f_res = dict(benchmark=f, score=f_score)
        f_res.update(record_losses)
        self.benchmark_scores.append(f_res)

    def cv_strategy(self, shuffle_split=False):
        if shuffle_split and self.stratify_cv_by is not None:
            cv = StratifiedShuffleSplit(
                n_splits=self.hypers["n_folds"],
                test_size=0.1,
                random_state=42,
            ).split(self.data, self.stratify_cv_by.reindex(self.data.samples))
        elif shuffle_split:
            cv = ShuffleSplit(
                n_splits=self.hypers["n_folds"], test_size=0.1, random_state=42
            ).split(self.data)
        elif self.stratify_cv_by is not None:
            cv = StratifiedKFold(
                n_splits=self.hypers["n_folds"], shuffle=True, random_state=42
            ).split(self.data, self.stratify_cv_by.reindex(self.data.samples))
        else:
            cv = KFold(
                n_splits=self.hypers["n_folds"], shuffle=True, random_state=42
            ).split(self.data)

        return cv

    def training(self, cv=None, drop_last=False, skip_cv_save=True):
        cv_idx, epoch = 0, 0

        cvtest_datasets = {n: [] for n in self.data.view_names}

        cv = self.cv_strategy() if cv is None else cv

        for cv_idx, (train_idx, test_idx) in enumerate(cv, start=1):
            is_early_stop = False

            # Train and Test Data
            data_train = torch.utils.data.Subset(self.data, train_idx)
            drop_last_train = bool(
                drop_last and len(data_train) >= int(self.hypers["batch_size"])
            )
            if drop_last and not drop_last_train:
                warnings.warn(
                    "drop_last=True but CV training split is smaller than batch_size; "
                    "using drop_last=False for this fold."
                )
            dataloader_train = DataLoader(
                data_train,
                batch_size=self.hypers["batch_size"],
                shuffle=True,
                drop_last=drop_last_train,
            )

            data_test = torch.utils.data.Subset(self.data, test_idx)
            dataloader_test = DataLoader(
                data_test, batch_size=self.hypers["batch_size"], shuffle=False
            )

            # Initialize Model and Optimizer
            model = self.initialize_model()
            model.to(self.device)
            optimizer = CLinesLosses.get_optimizer(model, self.hypers)
            scheduler = CLinesLosses.get_scheduler(optimizer, self.hypers)

            # Train and Test Model
            loss_previous, loss_counter = None, 0
            for epoch in range(1, self.hypers["num_epochs"] + 1):
                self.maybe_unfreeze_transfer_modules(model, epoch)

                # Train
                model.train()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_train,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="train",
                        lr=optimizer.param_groups[0]["lr"],
                    ),
                )

                # Test
                model.eval()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_test,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="val",
                        lr=optimizer.param_groups[0]["lr"],
                    ),
                )

                if self.hypers["model"] == "GMVAE" and self.hypers["gmvae_decay_temp"]:
                    new_gumbel_temp = np.maximum(
                        self.hypers["gmvae_init_temp"]
                        * np.exp(-self.hypers["gmvae_decay_temp_rate"] * epoch),
                        self.hypers["gmvae_min_temp"],
                    )

                    if self.verbose > 1:
                        print(f"Gumbel Temperature: {new_gumbel_temp:.3f}")

                    model.module.gumbel_temp = new_gumbel_temp

                self.print_losses(cv_idx, epoch)

                # Early Stopping
                loss_current = self.get_losses(cv_idx, epoch, "type").loc[
                    "val", "reconstruction"
                ]
                loss_current_total = self.get_losses(cv_idx, epoch, "type").loc[
                    "val", "total"
                ]

                # Check if loss is finite
                if not (np.isfinite(loss_current) and np.isfinite(loss_current_total)):
                    warnings.warn(f"NaN or Inf loss at cv {cv_idx}, epoch {epoch}.")
                    return np.nan, cvtest_datasets

                elif loss_previous is None:
                    loss_previous = loss_current

                elif round(loss_current, 4) < round(loss_previous, 4):
                    loss_counter = 0
                    loss_previous = loss_current

                else:
                    loss_counter += 1

                if loss_counter >= self.early_stop_patience:
                    warnings.warn(f"Early stopping at cv {cv_idx}, epoch {epoch}.")
                    is_early_stop = True

                if scheduler is not None:
                    self.update_learning_rate(scheduler, optimizer, loss_current, epoch)

                # If last epoch, save test predictions
                if epoch == self.hypers["num_epochs"] or is_early_stop:
                    data_test_all = DataLoader(
                        data_test, batch_size=len(test_idx), shuffle=False
                    )

                    for data in data_test_all:
                        x, y, _, x_mask = data

                        x = [m.to(self.device) for m in x]
                        x_masked = [
                            m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x)
                        ]
                        y = y.to(self.device)

                        with torch.no_grad():
                            if self.hypers["use_conditionals"]:
                                out_net = model(x_masked + [y])
                            else:
                                out_net = model(x_masked)

                        x_hat = out_net["x_hat"]

                        for name, df in zip(self.data.view_names, x_hat):
                            df_hat = pd.DataFrame(
                                self.data.view_scalers[name].inverse_transform(
                                    df.detach().cpu().numpy()
                                ),
                                index=pd.Series(self.data.samples)
                                .iloc[test_idx]
                                .values,
                                columns=self.data.view_feature_names[name],
                            )

                            if name in {"copynumber"}:
                                df_hat = df_hat.round()

                            cvtest_datasets[name].append(df_hat)

                    break

        # Concat test datasets
        cvtest_datasets = {
            name: pd.concat(dfs) for name, dfs in cvtest_datasets.items()
        }
        if not skip_cv_save:
            for name, df in cvtest_datasets.items():
                df.round(5).to_csv(
                    f"{plot_folder}/files/{self.timestamp}_imputed_{name}_cvtest.csv.gz",
                    compression="gzip",
                )

        return (
            self.get_losses(cv_idx, epoch, "type").loc["val", "reconstruction"],
            cvtest_datasets,
        )

    def update_learning_rate(self, scheduler, optimizer, loss_current, epoch):
        scheduler.step(loss_current)

        current_lr = optimizer.param_groups[0]["lr"]

        if round(current_lr, 4) < round(self.lrs[-1][1], 4):
            self.lrs.append((epoch, current_lr))

    def predictions(self, n_epochs=None, drop_last=False):
        imputed_datasets = dict()

        n_epochs = self.hypers["num_epochs"] if n_epochs is None else n_epochs

        # Data Loader
        drop_last_pred = bool(
            drop_last and len(self.data) >= int(self.hypers["batch_size"])
        )
        if drop_last and not drop_last_pred:
            warnings.warn(
                "drop_last=True but dataset is smaller than batch_size; "
                "using drop_last=False for predictions."
            )
        data_all = DataLoader(
            self.data,
            batch_size=self.hypers["batch_size"],
            shuffle=True,
            drop_last=drop_last_pred,
        )

        self.model = self.initialize_model()
        self.model.to(self.device)
        optimizer = CLinesLosses.get_optimizer(self.model, self.hypers)

        for e in range(1, n_epochs + 1):
            self.maybe_unfreeze_transfer_modules(self.model, e)
            self.model.train()
            print(f"Epoch {e:03}")
            self.epoch(
                self.model,
                optimizer,
                data_all,
            )

        # Make predictions and latent spaces
        data_all = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )

        self.model.eval()
        with torch.no_grad():
            for data in data_all:
                x, y, _, x_mask = data

                x = [m.to(self.device) for m in x]
                x_masked = [m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x)]
                y = y.to(self.device)

                if self.hypers["use_conditionals"]:
                    out_net = self.model(x_masked + [y])
                else:
                    out_net = self.model(x_masked)

                x_hat = out_net["x_hat"]
                z = out_net["z"]

                for name, df in zip(self.data.view_names, x_hat):
                    imputed_datasets[name] = pd.DataFrame(
                        self.data.view_scalers[name].inverse_transform(
                            df.detach().cpu().numpy()
                        ),
                        index=self.data.samples,
                        columns=self.data.view_feature_names[name],
                    )

                    if name in {"copynumber"}:
                        imputed_datasets[name] = imputed_datasets[name].round()

                z = pd.DataFrame(z.detach().cpu().numpy(), index=self.data.samples)
                z.columns = [f"Latent_{i+1}" for i in range(z.shape[1])]

                # Write to file
                for name, df in imputed_datasets.items():
                    df.round(5).to_csv(
                        f"{plot_folder}/files/{self.timestamp}_imputed_{name}.csv.gz",
                        compression="gzip",
                    )

                z.round(5).to_csv(
                    f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz",
                    compression="gzip",
                )

    def load_vae_reconstructions(self, mode="nans_only", dfs=None):
        """
        Load imputed data and latent space from files. "nans_only" mode, original
        measurements are mantained and only NaNs are imputed. "all" mode all
        data is imputed.

        Parameters
        ----------
        mode : str, optional
            Loading mode of imputed data, by default "nans_only"

        Returns
        -------
        dict
            Dictionary of imputed dataframes
            pandas.DataFrame
                Latent space

        Raises
        ------
        ValueError
            If mode is not "nans_only" or "all"

        """

        if mode not in ["nans_only", "all"]:
            raise ValueError(f"Invalid mode {mode}")

        if dfs is None:
            dfs = self.data.dfs

        dfs_imputed = {}
        for n in dfs:
            df_file = f"{plot_folder}/files/{self.timestamp}_imputed_{n}.csv.gz"

            if not os.path.isfile(df_file):
                continue

            df_imputed = pd.read_csv(df_file, index_col=0)

            if mode == "nans_only":
                df_imputed = self.data.dfs[n].combine_first(df_imputed)

            dfs_imputed[n] = df_imputed

        # Load latent space
        joint_latent = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz", index_col=0
        )

        return dfs_imputed, joint_latent

    def register_loss(self, loss, extra_fields=None):
        r = {
            k: np.round(float(v), 7)
            for k, v in loss.items()
            if type(v) == torch.Tensor and v.numel() == 1
        }

        if "reconstruction_views" in loss:
            for i, v in enumerate(loss["reconstruction_views"]):
                r[f"mse_{self.data.view_names[i]}"] = np.round(float(v), 7)

        if extra_fields is not None:
            r.update(extra_fields)

        self.losses.append(r)

    def get_losses(self, cv_idx, epoch_idx, groupby=None):
        l = pd.DataFrame(self.losses).query(f"cv == {cv_idx} & epoch == {epoch_idx}")
        if groupby is not None:
            l = l.groupby(groupby).mean()
        return l

    def get_benchmark(self, cv_idx, epoch_idx, groupby=None, benchmark=None):
        l = pd.DataFrame(self.benchmark_scores).query(
            f"cv == {cv_idx} & epoch == {epoch_idx}"
        )

        if benchmark is not None:
            l = l.query(f"benchmark == '{benchmark}'")

        if groupby is not None:
            l = l.groupby(groupby).mean()

        return l

    def print_single_loss(self, loss_dict, pbar=None):
        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] Loss "
        ptxt += f" | Total={loss_dict['total']:.2f}"

        for k in loss_dict:
            if k not in ["cv", "epoch", "type", "total", "lr"] and "_" not in k:
                ptxt += f" | {k}={loss_dict[k]:.2f}"

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def print_losses(self, cv_idx, epoch_idx, pbar=None):
        l = self.get_losses(cv_idx, epoch_idx, groupby="type")

        if l.empty:
            ptxt = (
                f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx:02}, "
                f"Epoch={epoch_idx:03} Loss unavailable"
            )
            if pbar is not None:
                pbar.set_description(ptxt)
            else:
                print(ptxt)
            return

        def _fmt_train_val(df, col):
            train_v = np.nan
            val_v = np.nan
            if "train" in df.index and col in df.columns:
                train_v = df.loc["train", col]
            if "val" in df.index and col in df.columns:
                val_v = df.loc["val", col]
            return f"{train_v:.2f}/{val_v:.2f}"

        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx:02}, Epoch={epoch_idx:03} Loss (train/val)"
        ptxt += f" | Total={_fmt_train_val(l, 'total')}"

        for k in l.columns:
            if k not in ["cv", "epoch", "type", "total", "lr"] and "_" not in k:
                ptxt += f" | {k}={_fmt_train_val(l, k)}"

        if self.verbose > 1:
            ptxt += f"\n[Benchmark scores (train/val)] "

            bench_df = self.get_benchmark(
                cv_idx, epoch_idx, groupby=["benchmark", "type"]
            ).reset_index()

            for b_name, b_df in bench_df.groupby("benchmark"):
                b_df = b_df.set_index("type")
                ptxt += f"{b_name}: {_fmt_train_val(b_df, 'score')} | "

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

    def load_losses_df(self):
        return pd.read_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv")

    def save_model(self):
        if self.model is None:
            warnings.warn("No model to save. Run predictions first.")
        else:
            torch.save(
                self.model.state_dict(),
                f"{plot_folder}/files/{self.timestamp}_model.pt",
            )

    def load_model(self):
        model_path = f"{plot_folder}/files/{self.timestamp}_model.pt"

        if not os.path.isfile(model_path):
            warnings.warn(f"No model to load. {model_path}")
            raise FileNotFoundError

        self.model = self.initialize_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def _get_shap_model(self):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        model_core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model_core, "base_mosa"):
            return model_core.base_mosa
        if hasattr(model_core, "base_transformer"):
            return model_core.base_transformer
        return model_core

    def _get_input_feature_names(self):
        feature_names_all = []
        view_names = list(self.data.view_names)
        for view_name in view_names:
            feature_names_all.append(
                self.data.features_mask[view_name][
                    self.data.features_mask[view_name] == True
                ].index.values
            )
        if self.hypers["use_conditionals"]:
            feature_names_all.append(self.data.labels_name)
            view_names = view_names + ["conditionals"]
        return view_names, feature_names_all

    def _get_output_feature_names(self, view_name):
        if self.hypers["filtered_encoder_only"]:
            return list(self.data.view_feature_names[view_name])
        return list(
            self.data.features_mask[view_name][
                self.data.features_mask[view_name] == True
            ].index
        )

    def _resolve_shap_targets(self, explain_target):
        if explain_target == "latent":
            return [f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])]

        if explain_target not in self.data.view_names:
            raise ValueError(f"Invalid explain_target '{explain_target}'.")

        all_features = self._get_output_feature_names(explain_target)
        target_features = all_features

        if explain_target == self.hypers.get("shap_target_view"):
            target_file = self.hypers.get("shap_target_genes_file")
            target_count = int(self.hypers.get("shap_target_gene_count", 50))
            all_features_set = set(all_features)
            file_genes = []

            if target_file is not None and os.path.isfile(target_file):
                file_genes = pd.read_csv(target_file, header=None).iloc[:, 0].astype(str)
                file_genes = file_genes.dropna().tolist()
                file_genes = [g for g in file_genes if g in all_features_set]
                file_genes = list(dict.fromkeys(file_genes))

            if explain_target == "crisprcas9":
                df = self.data.dfs[explain_target].reindex(columns=all_features)
                ess = (df < -0.5).sum().astype(float)
                var = df.var().astype(float)
                score = ess.rank(pct=True).fillna(0) + var.rank(pct=True).fillna(0)
                ranked_targets = score.sort_values(ascending=False).index.tolist()
            else:
                ranked_targets = (
                    self.data.dfs[explain_target]
                    .reindex(columns=all_features)
                    .var()
                    .sort_values(ascending=False)
                    .index.tolist()
                )

            if target_count == -1:
                target_features = ranked_targets
            elif target_count > 0:
                target_features = ranked_targets[:target_count]
            else:
                warnings.warn(
                    "shap_target_gene_count should be a positive integer or -1. "
                    f"Got {target_count}; defaulting to all targets."
                )
                target_features = ranked_targets

            # Always include explicit genes from file in addition to top-ranked targets.
            if len(file_genes) > 0:
                selected_genes = set(target_features)
                for gene in file_genes:
                    if gene not in selected_genes:
                        target_features.append(gene)
                        selected_genes.add(gene)

            if len(target_features) == 0:
                target_features = all_features

        return target_features

    def _resolve_shap_output_indices(self, explain_target, target_features):
        if explain_target == "latent":
            return None

        output_features = self._get_output_feature_names(explain_target)
        output_index = {f: i for i, f in enumerate(output_features)}
        return [output_index[f] for f in target_features if f in output_index]

    def _resolve_shap_sample_ids(self, n_samples):
        sample_ids = getattr(self, "_last_shap_sample_ids", None)
        if sample_ids is not None and len(sample_ids) == n_samples:
            return sample_ids

        all_samples = list(self.data.samples)
        if n_samples <= len(all_samples):
            return all_samples[:n_samples]

        return [f"sample_{i+1}" for i in range(n_samples)]

    @staticmethod
    def _ensure_shap_value_dims(shap_array):
        arr = np.asarray(shap_array)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.ndim != 3:
            raise ValueError(
                "Unexpected SHAP value shape. Expected 2D/3D array, "
                f"got shape={arr.shape}."
            )
        return arr

    @staticmethod
    def _chunk_indices(indices, chunk_size):
        chunk_size = max(1, int(chunk_size))
        return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    def _build_mean_abs_shap_df_from_values(self, shap_values, explain_target="latent"):
        view_names, feature_names_all = self._get_input_feature_names()
        n_targets = self._ensure_shap_value_dims(shap_values[0]).shape[-1]

        if explain_target == "latent":
            target_feature_names = [f"Latent_{i+1}" for i in range(n_targets)]
        else:
            target_feature_names = getattr(self, "_last_shap_target_names", None)
            if target_feature_names is None or len(target_feature_names) != n_targets:
                target_feature_names = self._resolve_shap_targets(explain_target)[:n_targets]

        all_target_df = []
        for i, view_name in enumerate(view_names):
            view_values = self._ensure_shap_value_dims(shap_values[i])
            mean_abs = np.abs(view_values).mean(axis=0)
            feature_names = [f"{view_name}_{c}" for c in feature_names_all[i]][: mean_abs.shape[0]]

            view_df = pd.DataFrame(
                mean_abs[:, : len(target_feature_names)].T,
                columns=feature_names,
            )
            all_target_df.append(view_df)

        if len(all_target_df) == 0:
            return pd.DataFrame(columns=["target_name"])

        shap_df = pd.concat(all_target_df, axis=1)
        shap_df.insert(0, "target_name", target_feature_names[: shap_df.shape[0]])

        if (
            self.hypers.get("shap_cross_omic_only", False)
            and explain_target == "crisprcas9"
        ):
            keep_cols = ["target_name"] + [
                c for c in shap_df.columns if c != "target_name" and not c.startswith("crisprcas9_")
            ]
            shap_df = shap_df[keep_cols]

        return shap_df

    def run_shap(
        self,
        n_samples=50,
        seed=42,
        explain_target="latent",
        use_all_samples=False,
        shap_batch_size=None,
        shap_grad_batch_size=None,
        use_data_parallel=False,
        target_chunk_size=None,
        show_progress=True,
        aggregate_abs_mean=False,
    ):
        torch.manual_seed(seed)
        if explain_target is None:
            explain_target = self.hypers.get("shap_target_view") or "latent"

        shap_model = self._get_shap_model()
        target_feature_names = self._resolve_shap_targets(explain_target)
        target_indices = self._resolve_shap_output_indices(
            explain_target, target_feature_names
        )

        shap_model.return_for_shap = explain_target
        shap_model.return_for_shap_indices = target_indices

        if target_indices is None:
            selected_target_names = list(target_feature_names)
        else:
            selected_target_names = target_feature_names[: len(target_indices)]

        if len(selected_target_names) == 0:
            raise ValueError(
                f"No SHAP targets resolved for explain_target='{explain_target}'."
            )

        if use_all_samples:
            batch_size = len(self.data.samples)
        elif shap_batch_size is not None:
            batch_size = max(1, int(shap_batch_size))
        elif explain_target in ["latent", "metabolomics", "drugresponse"]:
            batch_size = len(self.data.samples)
        elif explain_target in ["copynumber"]:
            batch_size = max(1, len(self.data.samples) // 5)
        elif explain_target in ["proteomics", "transcriptomics", "methylation"]:
            batch_size = max(1, len(self.data.samples) // 10)
        else:
            batch_size = 20

        forward_model = shap_model
        if use_data_parallel:
            if (
                isinstance(self.model, nn.DataParallel)
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 1
            ):
                forward_model = self.model
            else:
                warnings.warn(
                    "use_data_parallel=True but multi-GPU DataParallel is unavailable; "
                    "falling back to single-model SHAP."
                )

        self.model.eval()
        forward_model.eval()
        data_all = DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=False,
        )
        data = next(iter(data_all))
        x, y, _, x_mask = data

        x = [m.to(self.device) for m in x]
        x_masked = [m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x)]
        y = y.to(self.device)

        if (
            self.hypers.get("shap_cross_omic_only", False)
            and explain_target == "crisprcas9"
            and "crisprcas9" in self.data.view_names
        ):
            crispr_idx = self.data.view_names.index("crisprcas9")
            crispr_input = x_masked[crispr_idx]
            baseline = torch.nanmean(crispr_input, dim=0, keepdim=True)
            x_masked[crispr_idx] = baseline.repeat(crispr_input.shape[0], 1)

        n_shap_samples = int(x_masked[0].shape[0]) if len(x_masked) > 0 else 0
        self._last_shap_sample_ids = self._resolve_shap_sample_ids(n_shap_samples)

        input_list = x_masked + [y] if self.hypers["use_conditionals"] else x_masked

        # SHAP's PyTorch GradientExplainer calls model(*inputs), while MOSA-style
        # forward expects a single list argument. Adapt call signature here.
        class _ShapModelAdapter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, *model_inputs):
                if len(model_inputs) == 1 and isinstance(
                    model_inputs[0], (list, tuple)
                ):
                    model_inputs = tuple(model_inputs[0])
                return self.base_model(list(model_inputs))

        shap_model_adapter = _ShapModelAdapter(forward_model)
        shap_model_adapter.eval()

        view_names, feature_names_all = self._get_input_feature_names()
        if explain_target == "latent" or target_indices is None:
            index_chunks = [None]
            name_chunks = [selected_target_names]
        else:
            resolved_chunk_size = (
                int(target_chunk_size)
                if target_chunk_size is not None
                else int(self.hypers.get("shap_target_chunk_size", 1))
            )
            index_chunks = self._chunk_indices(target_indices, resolved_chunk_size)
            name_chunks = self._chunk_indices(selected_target_names, resolved_chunk_size)

        resolved_grad_batch_size = None
        if shap_grad_batch_size is not None:
            resolved_grad_batch_size = int(shap_grad_batch_size)
        elif self.hypers.get("shap_grad_batch_size") is not None:
            resolved_grad_batch_size = int(self.hypers.get("shap_grad_batch_size"))

        # `None` or non-positive values fall back to SHAP's default (50).
        if resolved_grad_batch_size is not None and resolved_grad_batch_size <= 0:
            resolved_grad_batch_size = None

        if show_progress:
            n_targets = len(selected_target_names)
            n_chunks = len(index_chunks)
            grad_batch_txt = (
                "default(50)"
                if resolved_grad_batch_size is None
                else str(resolved_grad_batch_size)
            )
            print(
                f"Running SHAP for {n_targets:,} target(s) in {n_chunks:,} chunk(s). "
                f"(aggregate_abs_mean={aggregate_abs_mean}, grad_batch_size={grad_batch_txt})"
            )

        chunk_iterator = zip(index_chunks, name_chunks)
        if show_progress and len(index_chunks) > 1:
            chunk_iterator = tqdm(
                chunk_iterator,
                total=len(index_chunks),
                desc=f"SHAP {explain_target} targets",
            )

        all_shap_df = []
        merged_explanation = None
        merged_values = None
        merged_base_values = None
        total_targets_seen = 0

        for chunk_indices, chunk_target_names in chunk_iterator:
            shap_model.return_for_shap_indices = chunk_indices
            # Re-create explainer per chunk because SHAP caches output dimensionality.
            if resolved_grad_batch_size is None:
                explainer = shap.explainers._gradient.GradientExplainer(
                    shap_model_adapter, input_list
                )
            else:
                explainer = shap.explainers._gradient.GradientExplainer(
                    shap_model_adapter,
                    input_list,
                    batch_size=resolved_grad_batch_size,
                )
            chunk_explanation = explainer(input_list, nsamples=n_samples)
            chunk_values = [self._ensure_shap_value_dims(v) for v in chunk_explanation.values]

            chunk_target_count = chunk_values[0].shape[-1]
            chunk_target_names = chunk_target_names[:chunk_target_count]
            total_targets_seen += chunk_target_count

            if aggregate_abs_mean:
                target_df = []
                for i, view_name in enumerate(view_names):
                    view_values = chunk_values[i]
                    mean_abs = np.abs(view_values).mean(axis=0)
                    feature_names = [f"{view_name}_{c}" for c in feature_names_all[i]][: mean_abs.shape[0]]
                    view_df = pd.DataFrame(
                        mean_abs[:, : len(chunk_target_names)].T,
                        columns=feature_names,
                    )
                    target_df.append(view_df)

                if len(target_df) > 0:
                    chunk_df = pd.concat(target_df, axis=1)
                    chunk_df.insert(0, "target_name", chunk_target_names[: chunk_df.shape[0]])

                    if (
                        self.hypers.get("shap_cross_omic_only", False)
                        and explain_target == "crisprcas9"
                    ):
                        keep_cols = ["target_name"] + [
                            c
                            for c in chunk_df.columns
                            if c != "target_name" and not c.startswith("crisprcas9_")
                        ]
                        chunk_df = chunk_df[keep_cols]

                    all_shap_df.append(chunk_df)
            else:
                if merged_explanation is None:
                    merged_explanation = chunk_explanation
                    merged_values = chunk_values
                    if getattr(chunk_explanation, "base_values", None) is not None:
                        merged_base_values = np.asarray(chunk_explanation.base_values)
                else:
                    merged_values = [
                        np.concatenate([merged_values[i], chunk_values[i]], axis=-1)
                        for i in range(len(merged_values))
                    ]
                    if (
                        merged_base_values is not None
                        and getattr(chunk_explanation, "base_values", None) is not None
                    ):
                        chunk_base_values = np.asarray(chunk_explanation.base_values)
                        if (
                            merged_base_values.ndim >= 1
                            and chunk_base_values.ndim >= 1
                            and merged_base_values.shape[:-1] == chunk_base_values.shape[:-1]
                        ):
                            merged_base_values = np.concatenate(
                                [merged_base_values, chunk_base_values], axis=-1
                            )

        if aggregate_abs_mean:
            if len(all_shap_df) == 0:
                warnings.warn("No SHAP rows generated.")
                self._last_shap_target_names = []
                return pd.DataFrame(columns=["target_name"])

            shap_df = pd.concat(all_shap_df, axis=0, ignore_index=True)
            self._last_shap_target_names = selected_target_names[:total_targets_seen]
            return shap_df

        if merged_explanation is None or merged_values is None:
            raise RuntimeError("SHAP execution produced no explanation output.")

        merged_explanation.values = merged_values
        merged_explanation.feature_names = feature_names_all
        if merged_base_values is not None:
            merged_explanation.base_values = merged_base_values

        self._last_shap_target_names = selected_target_names[: merged_values[0].shape[-1]]
        return merged_explanation

    def save_shap(self, shap_values, explain_target="latent"):
        if isinstance(shap_values, pd.DataFrame):
            shap_df = shap_values.copy()
        else:
            shap_df = self._build_mean_abs_shap_df_from_values(
                shap_values, explain_target=explain_target
            )

        required_cols = {"target_name", "omics_feature", "mean_abs_shap"}
        is_long_format = required_cols.issubset(set(shap_df.columns))
        if not is_long_format and "target_name" not in shap_df.columns:
            raise ValueError(
                "SHAP save expects either long-format columns "
                f"{sorted(required_cols)} or a wide table containing 'target_name'."
            )

        if is_long_format:
            if "omic_layer" not in shap_df.columns:
                shap_df["omic_layer"] = shap_df["omics_feature"].str.split("_").str[0]
            shap_df["mean_abs_shap"] = shap_df["mean_abs_shap"].astype(float)
        else:
            feature_cols = [
                c for c in shap_df.columns if c not in {"target_name", "Sample ID"}
            ]
            if len(feature_cols) > 0:
                shap_df[feature_cols] = shap_df[feature_cols].apply(
                    pd.to_numeric, errors="coerce"
                )

        is_aggregated_output = is_long_format or ("Sample ID" not in shap_df.columns)
        file_suffix = "_mean_abs" if is_aggregated_output else ""

        out_file = (
            f"{shap_folder}/files/"
            f"{self.timestamp}_shap_values_{explain_target}{file_suffix}.csv.gz"
        )
        shap_df.to_csv(out_file, compression="gzip", index=False)
        self.save_shap_rankings(shap_df, explain_target, file_suffix=file_suffix)
        return shap_df

    def save_shap_rankings(self, shap_df, explain_target="latent", file_suffix=""):
        if {"omics_feature", "mean_abs_shap"}.issubset(set(shap_df.columns)):
            feature_rank_df = (
                shap_df.groupby("omics_feature", as_index=False)["mean_abs_shap"]
                .mean()
                .rename(columns={"omics_feature": "feature", "mean_abs_shap": "importance"})
                .sort_values("importance", ascending=False)
            )
        else:
            feature_cols = [
                c for c in shap_df.columns if c not in {"target_name", "Sample ID"}
            ]
            if len(feature_cols) == 0:
                return
            feature_rank = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
            feature_rank_df = feature_rank.rename("importance").reset_index()
            feature_rank_df.columns = ["feature", "importance"]

        if "omic_layer" not in feature_rank_df.columns:
            feature_rank_df["omic_layer"] = feature_rank_df["feature"].str.split("_").str[0]

        feature_rank_df.to_csv(
            (
                f"{shap_folder}/files/"
                f"{self.timestamp}_shap_feature_ranking_{explain_target}{file_suffix}.csv"
            ),
            index=False,
        )

        omic_rank_df = (
            feature_rank_df.groupby("omic_layer", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        omic_rank_df.to_csv(
            (
                f"{shap_folder}/files/"
                f"{self.timestamp}_shap_omic_ranking_{explain_target}{file_suffix}.csv"
            ),
            index=False,
        )

        _, ax = plt.subplots(1, 1, figsize=(3, 1.6), dpi=300)
        sns.barplot(data=omic_rank_df, x="importance", y="omic_layer", color="#4c72b0", ax=ax)
        ax.set(
            xlabel="Mean absolute SHAP",
            ylabel="Omic layer",
            title=f"SHAP omic ranking ({explain_target})",
        )
        PhenPred.save_figure(
            f"{shap_folder}/{self.timestamp}_shap_omic_ranking_{explain_target}{file_suffix}"
        )

    def save_shap_top200_features(self, shap_values, explain_target="latent"):
        if isinstance(shap_values, pd.DataFrame):
            shap_df = shap_values.copy()
        else:
            shap_df = self._build_mean_abs_shap_df_from_values(
                shap_values, explain_target=explain_target
            )

        required_cols = {"target_name", "omics_feature", "mean_abs_shap"}
        is_aggregated_output = required_cols.issubset(set(shap_df.columns)) or (
            "Sample ID" not in shap_df.columns
        )
        file_suffix = "_mean_abs" if is_aggregated_output else ""
        if len(shap_df) == 0:
            warnings.warn("No SHAP rows to save after filtering.")
            return

        if required_cols.issubset(set(shap_df.columns)):
            if "omic_layer" not in shap_df.columns:
                shap_df["omic_layer"] = shap_df["omics_feature"].str.split("_").str[0]

            shap_top200_df = (
                shap_df.sort_values(
                    ["target_name", "mean_abs_shap"], ascending=[True, False]
                )
                .groupby("target_name", as_index=False)
                .head(200)
                .reset_index(drop=True)
            )
        else:
            if "target_name" not in shap_df.columns:
                raise ValueError(
                    "Top-200 SHAP export expects either aggregated long-format columns "
                    f"{sorted(required_cols)} or a wide table containing 'target_name'."
                )

            feature_cols = [
                c for c in shap_df.columns if c not in {"target_name", "Sample ID"}
            ]
            if len(feature_cols) == 0:
                warnings.warn("No SHAP feature columns found for top-200 export.")
                return

            top_rows = []
            row_iter = shap_df.iterrows()
            if len(shap_df) > 100:
                row_iter = tqdm(
                    row_iter,
                    total=len(shap_df),
                    desc=f"Top-200 {explain_target} targets",
                )
            for _, row in row_iter:
                target_name = row["target_name"]
                feature_vals = pd.to_numeric(row[feature_cols], errors="coerce").abs()
                top_vals = feature_vals.sort_values(ascending=False).head(200)
                if top_vals.empty:
                    continue

                top_df = pd.DataFrame(
                    {
                        "target_name": target_name,
                        "omics_feature": top_vals.index.astype(str),
                        "mean_abs_shap": top_vals.values,
                    }
                )
                top_rows.append(top_df)

            if len(top_rows) == 0:
                warnings.warn("No SHAP rows to save after top-200 selection.")
                return

            shap_top200_df = pd.concat(top_rows, axis=0, ignore_index=True)
            shap_top200_df["omic_layer"] = shap_top200_df["omics_feature"].str.split("_").str[0]

        print("Saving files...")
        shap_top200_df.to_feather(
            (
                f"{shap_folder}/files/"
                f"{self.timestamp}_shap_values_top_features_{explain_target}{file_suffix}.feather"
            )
        )
        return shap_top200_df

    def _plot_lr_rates(self, ax):
        for e, lr in self.lrs:
            ax.axvline(e, color="black", linestyle="--", alpha=0.5, lw=0.3)
            ax.text(
                e,
                ax.get_ylim()[1],
                f"LR={lr:.0e}",
                ha="left",
                va="top",
                rotation=90,
                fontsize=4,
            )

    def plot_losses(self, losses_df=None, loss_terms=None, figsize=(3, 2)):
        if losses_df is None:
            losses_df = self.load_losses_df()

        # Plot total losses
        plot_df = pd.melt(losses_df, id_vars=["epoch", "type"], value_vars="total")

        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="type",
            errorbar=("ci", 99),
            err_kws=dict(alpha=0.2, lw=0),
            ax=ax,
        )
        self._plot_lr_rates(ax)
        ax.set(
            title=f"Train and Validation Loss",
            xlabel="Epoch",
            ylabel="Loss",
        )
        ax.legend(
            title="Losses",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        PhenPred.save_figure(
            f"{plot_folder}/losses/{self.timestamp}_train_validation_loss"
        )

        # Plot reconstruction losses
        if loss_terms is None:
            cols = [
                c
                for c in losses_df
                if c not in ["cv", "epoch", "type", "total", "lr"] and "_" in c
            ]
        else:
            cols = loss_terms

        unique_prefix = {v.split("_")[0] for v in cols}
        for prefix in unique_prefix:
            plot_df = pd.melt(
                losses_df,
                id_vars=["epoch", "type"],
                value_vars=[c for c in cols if c.startswith(prefix)],
            )

            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
            sns.lineplot(
                data=plot_df,
                x="epoch",
                y="value",
                hue="variable",
                style="type",
                errorbar=("ci", 99),
                err_kws=dict(alpha=0.2, lw=0),
                ax=ax,
            )
            self._plot_lr_rates(ax)
            ax.legend(
                title="Losses",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
            ax.set(
                title=f"Total loss",
                xlabel="Epoch",
                ylabel="Loss",
            )
            PhenPred.save_figure(
                f"{plot_folder}/losses/{self.timestamp}_{prefix}_losses"
            )

        # Plot loss terms
        if loss_terms is None:
            cols = [
                c
                for c in losses_df
                if c not in ["cv", "epoch", "type", "total", "lr"] and "_" not in c
            ]
        else:
            cols = loss_terms

        plot_df = pd.melt(
            losses_df,
            id_vars=["epoch", "type"],
            value_vars=cols,
        )

        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="variable",
            style="type",
            errorbar=("ci", 99),
            err_kws=dict(alpha=0.2, lw=0),
            ax=ax,
        )
        self._plot_lr_rates(ax)
        ax.legend(
            title="Losses",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        ax.set(
            title=f"Total loss",
            xlabel="Epoch",
            ylabel="Loss",
        )
        PhenPred.save_figure(f"{plot_folder}/losses/{self.timestamp}_terms_losses")
