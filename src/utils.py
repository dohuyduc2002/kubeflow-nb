from pathlib import Path
import pandas as pd
import numpy as np
from optbinning import BinningProcess
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
from tqdm import tqdm
import yaml
from loguru import logger

CONFIG_PATH = (Path(__file__).parent.parent / "config.yaml").resolve()

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

USERNAME = config['kubeflow']['username']
PASSWORD = config['kubeflow']['password']
NAMESPACE = config['kubeflow']['namespace']
HOST = config['kubeflow']['host']

class Preprocess:
    def __init__(self, X_train, X_test, y_train, categorical_cols, numerical_cols):
        self.X_train_orig = X_train.copy()
        self.X_test_orig = X_test.copy()
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.X_train_processed = None
        self.X_test_processed = None
        self.binning_process = None

    def impute_data(self):
        for col in self.numerical_cols:
            if col in self.X_train.columns:
                median_val = self.X_train[col].median()
                self.X_train[col].fillna(median_val, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(median_val, inplace=True)

        for col in self.categorical_cols:
            if col in self.X_train.columns:
                mode_val = self.X_train[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "missing"
                self.X_train[col].fillna(fill_val, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(fill_val, inplace=True)

    def run(self):
        self.impute_data()
        all_features = self.categorical_cols + self.numerical_cols
        self.binning_process = BinningProcess(variable_names=all_features,
                                              categorical_variables=self.categorical_cols)
        self.binning_process.fit(self.X_train[all_features].to_numpy(), self.y_train)
        X_train_binned = self.binning_process.transform(self.X_train[all_features].to_numpy())
        self.X_train_processed = pd.DataFrame(X_train_binned, columns=all_features)

        if not self.X_test.empty:
            X_test_binned = self.binning_process.transform(self.X_test[all_features].to_numpy())
            self.X_test_processed = pd.DataFrame(X_test_binned, columns=all_features)
        else:
            self.X_test_processed = pd.DataFrame(columns=all_features)

        return self.X_train_processed, self.X_test_processed

    @staticmethod
    def compute_iv(series, y):
        df = pd.DataFrame({"bin": series, "target": y})
        total_good = (df["target"] == 0).sum()
        total_bad = (df["target"] == 1).sum()
        iv, eps = 0, 0.5
        for _, group in df.groupby("bin"):
            good = (group["target"] == 0).sum() or eps
            bad = (group["target"] == 1).sum() or eps
            dist_good = good / total_good
            dist_bad = bad / total_bad
            woe = np.log(dist_good / dist_bad)
            iv += (dist_good - dist_bad) * woe
        return iv

    def filter_features(self):
        if self.X_train_processed is None:
            raise RuntimeError("Processed data not available. Run the 'run()' method first.")

        features_to_exclude, iv_dict = [], {}
        for col in tqdm(self.X_train_processed.columns, desc="Filtering Features"):
            iv = Preprocess.compute_iv(self.X_train_processed[col], self.y_train)
            iv_dict[col] = iv
            missing_ratio = self.X_train_orig[col].isnull().mean()
            if iv > 0.5 or iv < 0.02 or missing_ratio > 0.1:
                features_to_exclude.append(col)

        self.X_train_processed.drop(columns=features_to_exclude, inplace=True)
        if self.X_test_processed is not None and not self.X_test_processed.empty:
            self.X_test_processed.drop(columns=features_to_exclude, errors='ignore', inplace=True)

        return iv_dict, features_to_exclude

class PreprocessFeatureSelector:
    def __init__(self, X_train, X_test, y_train, categorical_cols, numerical_cols,
                 data_version, save_train_data_path, save_test_data_path, n_features_to_select=None, fs_kwargs=None):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.data_version = data_version
        self.save_train_data_path = Path(save_train_data_path)
        self.save_test_data_path = Path(save_test_data_path)
        self.n_features_to_select = n_features_to_select or 'auto'
        self.fs_kwargs = fs_kwargs or {}
        self.preprocess_obj = None
        self.selected_features = []
        self.fs_excluded_features = []
        self.iv_excluded_features = []

    def run(self):
        logger.info("🚀 Starting preprocessing and filtering...")
        self.preprocess_obj = Preprocess(self.X_train, self.X_test, self.y_train,
                                         self.categorical_cols, self.numerical_cols)
        X_train_proc, X_test_proc = self.preprocess_obj.run()
        logger.info("✅ Preprocessing complete.")

        logger.info("🧮 Features before filtering: {}", list(X_train_proc.columns))
        iv_dict, iv_excluded = self.preprocess_obj.filter_features()
        self.iv_excluded_features.extend(iv_excluded)

        logger.info("🧹 Filtering complete.")
        logger.info("📊 IV Values: {}", iv_dict)
        logger.info("❌ Features excluded during filtering: {}", iv_excluded)

        filtered_features = list(self.preprocess_obj.X_train_processed.columns)
        logger.info("🔍 Starting Sequential Feature Selection (SFS)...")

        sfs = SequentialFeatureSelector(XGBClassifier(eval_metric='auc'),
                                        n_features_to_select=self.n_features_to_select,
                                        direction='forward',
                                        **self.fs_kwargs)
        sfs.fit(self.preprocess_obj.X_train_processed, self.y_train)

        self.selected_features = list(self.preprocess_obj.X_train_processed.columns[sfs.get_support()])
        self.fs_excluded_features = list(set(filtered_features) - set(self.selected_features))

        logger.info("✅ SFS complete.")
        logger.info("🏁 Final selected features: {}", self.selected_features)
        logger.info("🚫 Features excluded by SFS: {}", self.fs_excluded_features)

        selected_train = pd.DataFrame(sfs.transform(self.preprocess_obj.X_train_processed),
                                      columns=self.selected_features)
        selected_train["TARGET"] = self.y_train

        selected_test = None
        if self.preprocess_obj.X_test_processed is not None and not self.preprocess_obj.X_test_processed.empty:
            selected_test = pd.DataFrame(sfs.transform(self.preprocess_obj.X_test_processed),
                                         columns=self.selected_features)

        combined_excluded = self.iv_excluded_features + self.fs_excluded_features

        self.save_train_data_path.mkdir(parents=True, exist_ok=True)
        self.save_test_data_path.mkdir(parents=True, exist_ok=True)

        train_filename = self.save_train_data_path / f"processed_train_{self.data_version}.csv"
        test_filename = self.save_test_data_path / f"processed_test_{self.data_version}.csv"

        selected_train.to_csv(train_filename, index=False)
        if selected_test is not None:
            selected_test.to_csv(test_filename, index=False)

        logger.info("💾 Saved processed training data to {}", train_filename)
        logger.info("💾 Saved processed test data to {}", test_filename)

        return selected_train, selected_test, self.selected_features, combined_excluded
