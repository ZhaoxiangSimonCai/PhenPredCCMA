import os
import json
import warnings
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm, zscore
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer, StandardScaler

from PhenPred.Utils import scale
from PhenPred.vae import data_folder


class CLinesDatasetCCMA(Dataset):
    def __init__(
        self,
        datasets,
        labels_names=None,
        decimals=4,
        feature_miss_rate_thres=0.9,
        standardize=False,
        normalize_features=False,
        normalize_samples=False,
        filter_features=None,
        filtered_encoder_only=False,
        min_views_per_sample=2,
        align_to_reference_features=False,
        reference_hypers_json=None,
        reference_feature_views=None,
        labels_mutations_file=None,
    ):
        super().__init__()

        self.labels_names = labels_names or []
        self.datasets = datasets
        self.decimals = decimals
        self.feature_miss_rate_thres = feature_miss_rate_thres
        self.standardize = standardize
        self.normalize_features = normalize_features
        self.normalize_samples = normalize_samples
        self.filter_features = filter_features or []
        self.filtered_encoder_only = filtered_encoder_only
        self.min_views_per_sample = max(1, int(min_views_per_sample))
        self.align_to_reference_features = bool(align_to_reference_features)
        self.reference_hypers_json = reference_hypers_json
        self.reference_feature_views = reference_feature_views
        self.labels_mutations_file = labels_mutations_file
        self._external_mutations = None

        self.dfs = {n: pd.read_csv(f, index_col=0) for n, f in self.datasets.items()}
        self.dfs = {
            n: df if n in {"crisprcas9", "copynumber"} else df.T
            for n, df in self.dfs.items()
        }

        if "crisprcas9" in self.dfs:
            self.dfs["crisprcas9"].columns = (
                self.dfs["crisprcas9"].columns.astype(str).str.split(" ").str[0]
            )
            self.dfs["crisprcas9"] = scale(self.dfs["crisprcas9"].T).T

        self._load_external_mutations_labels()

        if self.align_to_reference_features:
            self._apply_reference_feature_filter()

        self._remove_features_missing_values()
        self._build_samplesheet()
        self._samples_union()
        self._features_mask()

        if self.normalize_samples:
            self.dfs = {
                n: df if n in {"copynumber", "mutations"} else self.normalize_dataset(df)
                for n, df in self.dfs.items()
            }

        self._standardize_dfs()
        self._derive_optional_tables()
        self._build_labels()

        self.x_mask = [
            torch.tensor(self.features_mask[n].values, dtype=torch.bool)
            for n in self.views
        ]

        self.view_name_map = dict(
            copynumber="Copy number",
            mutations="Mutations",
            fusions="Fusions",
            methylation="Methylation",
            transcriptomics="Transcriptomics",
            proteomics="Proteomics",
            phosphoproteomics="Phosphoproteomics",
            metabolomics="Metabolomics",
            drugresponse="Drug response",
            crisprcas9="CRISPR-Cas9",
            growth="Growth",
        )

        print(self)

    def __str__(self):
        txt = f"CCMA | Samples = {len(self.samples):,}"
        for n, df in self.dfs.items():
            f_masked = df.shape[1] - int(self.features_mask[n].sum())
            view_name = self.view_name_map.get(n, n)
            txt += f" | {view_name} = {df.shape[1]:,} ({f_masked:,} masked)"
        txt += f" | Labels = {self.labels_size:,}"
        return txt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = [df[idx] for df in self.views.values()]
        x_nans = [df[idx] for df in self.view_nans.values()]
        y = self.labels[idx]
        return x, y, x_nans, self.x_mask

    def _remove_features_missing_values(self):
        for n in self.dfs:
            miss_rate = self.dfs[n].isnull().mean()
            self.dfs[n] = self.dfs[n].loc[
                :, miss_rate < float(self.feature_miss_rate_thres)
            ]

    def _load_external_mutations_labels(self):
        if self.labels_mutations_file is None:
            return

        mutations_path = self._resolve_reference_path(self.labels_mutations_file)
        if mutations_path is None:
            warnings.warn(
                f"Could not resolve labels_mutations_file '{self.labels_mutations_file}'."
            )
            return

        mut = pd.read_csv(mutations_path, index_col=0)
        mut.index = mut.index.astype(str)
        mut.columns = mut.columns.astype(str)

        sample_ids = set().union(
            *[set(df.index.astype(str).tolist()) for df in self.dfs.values()]
        )
        overlap_index = len(sample_ids.intersection(set(mut.index)))
        overlap_cols = len(sample_ids.intersection(set(mut.columns)))

        if overlap_cols > overlap_index:
            mut = mut.T

        mut.index = mut.index.astype(str)
        mut.columns = mut.columns.astype(str).str.replace("_snv$", "", regex=True)
        mut = mut.loc[:, ~mut.columns.duplicated()]
        mut = mut.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self._external_mutations = mut

    @staticmethod
    def _unique_preserve_order(values):
        out, seen = [], set()
        for value in values:
            value = str(value)
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _resolve_reference_path(self, path):
        if path is None:
            return None

        if os.path.isabs(path) and os.path.isfile(path):
            return path

        candidates = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(data_folder, path),
        ]

        if self.reference_hypers_json is not None:
            candidates.append(os.path.join(os.path.dirname(self.reference_hypers_json), path))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return None

    def _read_reference_view_features(self, file_path, view_name):
        if view_name in {"crisprcas9", "copynumber"}:
            features = pd.read_csv(file_path, index_col=0, nrows=0).columns.tolist()
        else:
            features = (
                pd.read_csv(file_path, index_col=0, usecols=[0]).index.astype(str).tolist()
            )

        if view_name == "crisprcas9":
            features = [str(v).split(" ")[0] for v in features]

        return self._unique_preserve_order(features)

    def _apply_reference_feature_filter(self):
        if self.reference_hypers_json is None:
            warnings.warn(
                "align_to_reference_features=True but reference_hypers_json is missing; "
                "skipping reference feature filtering."
            )
            return

        reference_hypers_path = self._resolve_reference_path(self.reference_hypers_json)
        if reference_hypers_path is None:
            warnings.warn(
                f"Could not resolve reference_hypers_json '{self.reference_hypers_json}'; "
                "skipping reference feature filtering."
            )
            return

        with open(reference_hypers_path, "r") as f:
            reference_hypers = json.load(f)

        reference_datasets = reference_hypers.get("datasets", {})
        if not isinstance(reference_datasets, dict) or len(reference_datasets) == 0:
            warnings.warn(
                "Reference hyperparameters do not contain datasets; "
                "skipping reference feature filtering."
            )
            return

        if self.reference_feature_views is not None:
            selected_views = [v for v in self.reference_feature_views if v in self.dfs]
        else:
            selected_views = [v for v in self.dfs if v in reference_datasets]

        if len(selected_views) == 0:
            warnings.warn(
                "No overlapping views between current CCMA datasets and reference datasets; "
                "skipping reference feature filtering."
            )
            return

        for view_name in selected_views:
            if view_name not in reference_datasets:
                continue

            raw_path = reference_datasets[view_name]
            ref_path = self._resolve_reference_path(raw_path)
            if ref_path is None:
                warnings.warn(
                    f"Could not resolve reference dataset path for view '{view_name}': "
                    f"'{raw_path}'. Skipping this view."
                )
                continue

            reference_features = self._read_reference_view_features(ref_path, view_name)
            current_df = self.dfs[view_name]

            shared_features = [f for f in reference_features if f in current_df.columns]
            if len(shared_features) == 0:
                raise ValueError(
                    f"Reference feature filter removed all features for view '{view_name}'. "
                    "No overlapping feature names between CCMA and reference dataset."
                )

            old_n = current_df.shape[1]
            self.dfs[view_name] = current_df.reindex(columns=shared_features)
            print(
                f"CCMA reference feature filter | {view_name}: "
                f"{old_n:,} -> {self.dfs[view_name].shape[1]:,} features"
            )

    def _build_samplesheet(self):
        all_ids = sorted(
            set().union(*[set(df.index.astype(str).tolist()) for df in self.dfs.values()])
        )
        self.samplesheet = pd.DataFrame(index=all_ids)
        self.samplesheet["model_id"] = self.samplesheet.index
        self.samplesheet["BROAD_ID"] = np.nan
        self.samplesheet["tissue"] = "Unknown"
        self.samplesheet["cancer_type"] = "Unknown"
        self.samplesheet["growth_properties_sanger"] = "Unknown"
        self.samplesheet["growth_properties_broad"] = "Unknown"

        # Keep these attributes for compatibility with existing benchmark code paths.
        self.ss_cmp = pd.DataFrame(index=self.samplesheet.index)
        self.ss_cmp["msi_status"] = np.nan

    def _samples_union(self):
        counts = pd.concat(
            [pd.Series(df.index.astype(str)) for df in self.dfs.values()], axis=0
        ).value_counts()

        self.samples = counts[counts >= self.min_views_per_sample].index
        self.samples = sorted(
            list(set(self.samples).intersection(set(self.samplesheet.index)))
        )

        if len(self.samples) == 0:
            raise ValueError(
                "No CCMA samples passed min_views_per_sample. "
                f"Configured min_views_per_sample={self.min_views_per_sample}; "
                f"available views={list(self.dfs.keys())}."
            )

        self.dfs = {n: df.reindex(index=self.samples) for n, df in self.dfs.items()}

    def _features_mask(self):
        self.features_mask = {}
        for n, df in self.dfs.items():
            mask = pd.Series(np.ones(df.shape[1], dtype=bool), index=df.columns)

            if n in self.filter_features:
                if n == "crisprcas9":
                    mask = (df < -0.5).sum() > 0
                elif n == "copynumber":
                    mask = (df.abs() == 2).sum() > 3
                elif n == "mutations":
                    mask = (df.fillna(0) > 0).sum() > 0
                else:
                    thres = self.gaussian_mixture_std(df)
                    mask = df.std() > thres

            self.features_mask[n] = mask

    def _standardize_dfs(self):
        self.views = {}
        self.view_scalers = {}
        self.view_feature_names = {}
        self.view_nans = {}
        self.view_names = []

        for n, df in self.dfs.items():
            x, scaler, x_nan = self.process_df(n, df)
            self.views[n] = x
            self.view_scalers[n] = scaler
            self.view_nans[n] = x_nan
            self.view_feature_names[n] = list(df.columns)
            self.view_names.append(n)

    def _derive_optional_tables(self):
        self.mutations = pd.DataFrame(index=self.samples)
        if "mutations" in self.dfs:
            self.mutations = self.dfs["mutations"].fillna(0).astype(float)
        elif self._external_mutations is not None:
            self.mutations = (
                self._external_mutations.reindex(index=self.samples).fillna(0).astype(float)
            )

        self.fusions = pd.DataFrame(index=self.samples)
        self.cnv = pd.DataFrame(index=self.samples)
        if "copynumber" in self.dfs:
            self.cnv = self.dfs["copynumber"].copy()

        self.growth = pd.DataFrame(
            index=self.samples,
            columns=["day4_day1_ratio", "doubling_time_hours"],
            data=np.nan,
        )

        self.drug_targets = pd.Series(dtype=object)

    def _build_labels(self, min_obs=0):
        labels = []

        if "tissue" in self.labels_names:
            labels.append(pd.get_dummies(self.samplesheet["tissue"]).add_prefix("tissue_"))

        if "cancer_type" in self.labels_names:
            labels.append(pd.get_dummies(self.samplesheet["cancer_type"]))

        if "culture" in self.labels_names:
            labels.append(
                pd.concat(
                    [
                        pd.get_dummies(self.samplesheet["growth_properties_broad"]),
                        pd.get_dummies(self.samplesheet["growth_properties_sanger"]),
                    ],
                    axis=1,
                )
            )

        if "growth" in self.labels_names:
            growth_df = self.growth[["day4_day1_ratio", "doubling_time_hours"]]
            growth_df = pd.DataFrame(
                zscore(growth_df, nan_policy="omit"),
                index=growth_df.index,
                columns=growth_df.columns,
            )
            labels.append(growth_df)

        if "mutations" in self.labels_names and not self.mutations.empty:
            labels.append(
                self.mutations.loc[:, self.mutations.sum() >= min_obs].add_prefix("mut_")
            )
        elif "mutations" in self.labels_names and self.mutations.empty:
            warnings.warn(
                "labels include 'mutations' but no mutation table was available; "
                "using fallback constant label."
            )

        if len(labels) == 0:
            self.labels = pd.DataFrame(
                np.ones((len(self.samples), 1)), index=self.samples, columns=["ones"]
            )
        else:
            self.labels = pd.concat(labels, axis=1)
            self.labels = self.labels.reindex(index=self.samples).fillna(0)

        self.labels_name = self.labels.columns.tolist()
        self.labels_size = self.labels.shape[1]
        self.labels = torch.tensor(self.labels.values.astype(float), dtype=torch.float)

    def process_df(self, df_name, df):
        to_standardize = (
            True
            if df_name not in {"copynumber", "mutations"} and self.standardize
            else False
        )

        if self.normalize_features:
            scaler = PowerTransformer(method="yeo-johnson", standardize=to_standardize)
        else:
            scaler = StandardScaler(with_mean=to_standardize, with_std=to_standardize)

        x = scaler.fit_transform(df).round(self.decimals)
        x_nan = ~np.isnan(x)

        if df_name in {"copynumber", "mutations"}:
            x[~x_nan] = 0
        else:
            x[~x_nan] = np.nanmean(x)

        x = torch.tensor(x, dtype=torch.float)
        return x, scaler, x_nan

    def normalize_dataset(self, df):
        l2_norms = np.sqrt(np.nansum(df**2, axis=1))
        return df / l2_norms[:, np.newaxis]

    def gaussian_mixture_std(self, df):
        df_std = df.std(axis=0).dropna()
        if df_std.empty or df_std.nunique() <= 1:
            return 0.0

        gm = GaussianMixture(n_components=2, random_state=0).fit(df_std.to_frame())
        gm_means = gm.means_.reshape(-1)
        gm_std = np.sqrt(gm.covariances_.reshape(-1))

        def solve(m1, m2, std1, std2):
            a = 1 / (2 * std1**2) - 1 / (2 * std2**2)
            b = m2 / (std2**2) - m1 / (std1**2)
            c = m1**2 / (2 * std1**2) - m2**2 / (2 * std2**2) - np.log(std2 / std1)
            return np.roots([a, b, c])

        intersections = solve(gm_means[0], gm_means[1], gm_std[0], gm_std[1])
        intersections = [x.real for x in intersections if np.isreal(x)]
        return float(max(intersections)) if intersections else 0.0

    def get_view_feature_index(self, feature_name, view_name):
        return self.view_feature_names[view_name].index(feature_name)

    def get_view_feature_by_name(self, feature_name, view_name):
        return self.views[view_name][:, self.get_view_feature_index(feature_name, view_name)]

    def samples_by_tissue(self, tissue):
        return (
            (self.samplesheet["tissue"] == tissue)
            .reindex(self.samples)
            .fillna(False)
            .astype(int)
            .rename(tissue)
        )

    def get_features(self, view_features_dict, dfs=None):
        if dfs is None:
            dfs = self.dfs

        return pd.concat(
            [
                dfs[v].reindex(columns=f).add_suffix(f"_{v}")
                for v, f in view_features_dict.items()
                if v in dfs
            ],
            axis=1,
        )
