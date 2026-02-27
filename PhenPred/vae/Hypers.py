import os
import json
import warnings
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Losses import CLinesLosses


class Hypers:
    @classmethod
    def read_json(cls, json_file):
        with open(json_file, "r") as f:
            hypers = json.load(f)
        return hypers

    @classmethod
    def read_hyperparameters(
        cls, hypers_json=None, parse_torch_functions=True, timestamp=None
    ):
        if timestamp is not None:
            hypers_json = f"{plot_folder}/files/{timestamp}_hyperparameters.json"
        elif hypers_json is None:
            hypers_json = f"{plot_folder}/files/hyperparameters.json"

        hypers = cls.read_json(hypers_json)

        if timestamp is not None:
            hypers["load_run"] = timestamp

        hypers.setdefault("model", "MOSA")
        hypers.setdefault("standardize", False)
        hypers.setdefault("w_rec", 1)
        hypers.setdefault("w_gauss", 0.01)
        hypers.setdefault("w_cat", 0.01)
        hypers.setdefault("labels", [])
        hypers.setdefault("filter_features", [])
        hypers.setdefault("filtered_encoder_only", False)
        hypers.setdefault("feature_miss_rate_thres", 0.9)
        hypers.setdefault("skip_benchmarks", False)
        hypers.setdefault("benchmark_mode", "full")
        hypers.setdefault("dataset_class", "depmap24q4")
        hypers.setdefault("transfer_mode", None)
        hypers.setdefault("transfer_checkpoint", None)
        hypers.setdefault("transfer_hypers_json", None)
        hypers.setdefault("transfer_freeze_epochs", 0)
        hypers.setdefault("min_views_per_sample", 2)
        hypers.setdefault("labels_mutations_file", None)
        hypers.setdefault("align_to_reference_features", False)
        hypers.setdefault("reference_hypers_json", None)
        hypers.setdefault("reference_feature_views", None)
        hypers.setdefault("shap_target_view", None)
        hypers.setdefault("shap_target_gene_count", 50)
        hypers.setdefault("shap_target_genes_file", None)
        hypers.setdefault("shap_cross_omic_only", False)
        hypers.setdefault("shap_target_chunk_size", 1)
        hypers.setdefault("shap_grad_batch_size", None)
        hypers.setdefault("grad_clip_max_norm", None)
        hypers.setdefault("kl_logvar_clip_min", None)
        hypers.setdefault("kl_logvar_clip_max", None)

        if hypers.get("use_conditionals") is None:
            hypers["use_conditionals"] = True

        n_views = len(hypers["datasets"])
        if hypers.get("view_loss_recon_type") is None:
            hypers["view_loss_recon_type"] = [
                "macro" if v == "copynumber" else "mean"
                for v in hypers["datasets"].keys()
            ]

        if len(hypers["view_loss_recon_type"]) != n_views:
            raise ValueError(
                "Invalid hyperparameters: `view_loss_recon_type` length must match "
                f"`datasets` length. Got {len(hypers['view_loss_recon_type'])} vs {n_views}."
            )

        if hypers.get("view_loss_weights") is None:
            hypers["view_loss_weights"] = [1.0] * n_views

        if len(hypers["view_loss_weights"]) != n_views:
            raise ValueError(
                "Invalid hyperparameters: `view_loss_weights` length must match "
                f"`datasets` length. Got {len(hypers['view_loss_weights'])} vs {n_views}."
            )

        if timestamp is None:  # full path is already stored in previous json config
            hypers["datasets"] = {
                k: (v if os.path.isabs(v) else f"{data_folder}/{v}")
                for k, v in hypers["datasets"].items()
            }
            if hypers.get("labels_mutations_file") is not None and not os.path.isabs(
                hypers["labels_mutations_file"]
            ):
                hypers["labels_mutations_file"] = (
                    f"{data_folder}/{hypers['labels_mutations_file']}"
                )

        print(f"# ---- Hyperparameters")
        print(json.dumps(hypers, indent=4, sort_keys=True))

        if parse_torch_functions:
            hypers = cls.parse_torch_functions(hypers)

        return hypers

    @classmethod
    def parse_torch_functions(cls, hypers):
        if hypers.get("activation_function") in {None, "<not serializable>"}:
            warnings.warn(
                "activation_function was not serializable in saved hyperparameters; "
                "falling back to 'prelu'."
            )
            hypers["activation_function"] = "prelu"

        if hypers.get("reconstruction_loss") in {None, "<not serializable>"}:
            warnings.warn(
                "reconstruction_loss was not serializable in saved hyperparameters; "
                "falling back to 'mse'."
            )
            hypers["reconstruction_loss"] = "mse"

        if (
            type(hypers["activation_function"]) == str
            and hypers["activation_function"] != "<not serializable>"
        ):
            hypers["activation_function"] = CLinesLosses.activation_function(
                hypers["activation_function"]
            )

        if (
            type(hypers["reconstruction_loss"]) == str
            and hypers["reconstruction_loss"] != "<not serializable>"
        ):
            hypers["reconstruction_loss"] = CLinesLosses.reconstruction_loss_method(
                hypers["reconstruction_loss"]
            )

        if type(hypers["hidden_dims"]) == str:
            hypers["hidden_dims"] = [
                float(l.strip()) for l in hypers["hidden_dims"].split(",")
            ]

        return hypers
