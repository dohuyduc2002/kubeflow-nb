import kfp
from kfp import dsl 

@dsl.component(
    base_image="microwave1005/scipy-img:latest"
)
def preprocess_binning_component(
    train_csv: dsl.Input[dsl.Dataset],
    test_csv : dsl.Input[dsl.Dataset],
    categorical_cols_json: dsl.Input[dsl.Artifact],
    numerical_cols_json : dsl.Input[dsl.Artifact],
    processed_train: dsl.Output[dsl.Dataset],
    processed_test : dsl.Output[dsl.Dataset],
    transformer_joblib: dsl.Output[dsl.Artifact],
    target_col: str,
    data_version: str = "v1",
):

    import pandas as pd
    import numpy as np
    import json, joblib
    from pathlib import Path
    from optbinning import BinningProcess
    from sklearn.feature_selection import SequentialFeatureSelector
    from xgboost import XGBClassifier
    from tqdm import tqdm
    from loguru import logger

    df_train = pd.read_csv(train_csv.path)
    df_test  = pd.read_csv(test_csv.path)

    with open(categorical_cols_json.path) as f_cat, open(numerical_cols_json.path) as f_num:
        categorical_cols = json.load(f_cat)
        numerical_cols   = json.load(f_num)

    # ---------- build X / y ----------
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col])
    else:
        X_test = df_test.copy()

    # ------------------------------------------------------------------
    #   Helper classes (trimmed but identical logic to your code)
    # ------------------------------------------------------------------
    class Preprocess:
        def __init__(self, X_tr, X_te, y_tr, cat_cols, num_cols):
            self.X_train_orig = X_tr.copy()
            self.X_test_orig  = X_te.copy()
            self.X_train      = X_tr.copy()
            self.X_test       = X_te.copy()
            self.y_train      = y_tr
            self.categorical_cols = cat_cols
            self.numerical_cols   = num_cols
            self.X_train_processed = None
            self.X_test_processed  = None
            self.binning_process   = None

        def impute_data(self):
            for col in self.numerical_cols:
                med = self.X_train[col].median()
                self.X_train[col].fillna(med, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(med, inplace=True)

            for col in self.categorical_cols:
                mode_val = self.X_train[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "missing"
                self.X_train[col].fillna(fill_val, inplace=True)
                if col in self.X_test.columns:
                    self.X_test[col].fillna(fill_val, inplace=True)

        def run(self):
            self.impute_data()
            all_feats = self.categorical_cols + self.numerical_cols
            self.binning_process = BinningProcess(
                variable_names=all_feats, categorical_variables=self.categorical_cols
            )
            self.binning_process.fit(self.X_train[all_feats].to_numpy(), self.y_train)
            self.X_train_processed = pd.DataFrame(
                self.binning_process.transform(self.X_train[all_feats].to_numpy()),
                columns=all_feats,
            )
            if not self.X_test.empty:
                self.X_test_processed = pd.DataFrame(
                    self.binning_process.transform(self.X_test[all_feats].to_numpy()),
                    columns=all_feats,
                )
            else:
                self.X_test_processed = pd.DataFrame(columns=all_feats)
            return self.X_train_processed, self.X_test_processed

        @staticmethod
        def compute_iv(series, y):
            df = pd.DataFrame({"bin": series, "target": y})
            tot_gd = (df["target"] == 0).sum()
            tot_bd = (df["target"] == 1).sum()
            iv, eps = 0, 0.5
            for _, grp in df.groupby("bin"):
                gd = (grp["target"] == 0).sum() or eps
                bd = (grp["target"] == 1).sum() or eps
                iv += ((gd / tot_gd) - (bd / tot_bd)) * np.log((gd / tot_gd) / (bd / tot_bd))
            return iv

        def filter_features(self):
            excl, iv_dict = [], {}
            for col in tqdm(self.X_train_processed.columns, desc="Filtering"):
                iv_val = Preprocess.compute_iv(self.X_train_processed[col], self.y_train)
                iv_dict[col] = iv_val
                miss = self.X_train_orig[col].isnull().mean()
                if iv_val > 0.5 or iv_val < 0.02 or miss > 0.10:
                    excl.append(col)
            self.X_train_processed.drop(columns=excl, inplace=True)
            self.X_test_processed .drop(columns=excl, errors="ignore", inplace=True)
            return iv_dict, excl

    logger.info(" Preprocessing …")
    pre = Preprocess(X_train, X_test, y_train, categorical_cols, numerical_cols)
    X_train_proc, X_test_proc = pre.run()
    iv_dict, iv_excl = pre.filter_features()

    logger.info(" Sequential Feature Selection …")
    sfs = SequentialFeatureSelector(
        XGBClassifier(eval_metric="auc"),
        n_features_to_select="auto",
        direction="forward",
    )
    sfs.fit(X_train_proc, y_train)
    selected_cols = list(X_train_proc.columns[sfs.get_support()])

    final_train = pd.DataFrame(sfs.transform(X_train_proc), columns=selected_cols)
    final_train[target_col] = y_train

    final_test = (
        pd.DataFrame(sfs.transform(X_test_proc), columns=selected_cols)
        if not X_test_proc.empty else pd.DataFrame(columns=selected_cols)
    )

    # ------------------------------------------------------------------
    #  Write outputs
    # ------------------------------------------------------------------
    final_train.to_csv(processed_train.path, index=False)
    final_test .to_csv(processed_test.path , index=False)

    # Save the fitted artefacts
    joblib.dump(
        {
            "binning_process": pre.binning_process,
            "sfs_selector"  : sfs,
            "selected_cols" : selected_cols,
            "iv_excluded"   : iv_excl,
        },
        transformer_joblib.path,
    )

    logger.info(" Preprocessing + binning complete.")
