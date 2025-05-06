import kfp
from kfp import dsl 

@dsl.component(
    base_image="microwave1005/scipy-img:latest",
    packages_to_install=["jsonschema"],
)
def detect_feature_types_component(
    train_csv: dsl.Input[dsl.Dataset],
    categorical_cols_json: dsl.Output[dsl.Artifact],
    numerical_cols_json: dsl.Output[dsl.Artifact],
):
    """
    Reads the training CSV and outputs two JSON files:
    1. categorical_cols.json
    2. numerical_cols.json
    """

    # All non-KFP imports live inside the function
    import pandas as pd
    import json

    # ------------------------------------------------------------------
    #  Load training data
    # ------------------------------------------------------------------
    df = pd.read_csv(train_csv.path)

    # Identify column types
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove identifiers / target
    for col in ["SK_ID_CURR", "TARGET"]:
        if col in numerical_features:
            numerical_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)

    # ------------------------------------------------------------------
    #  Persist lists as JSON
    # ------------------------------------------------------------------
    with open(categorical_cols_json.path, "w") as f_cat, open(
        numerical_cols_json.path, "w"
    ) as f_num:
        json.dump(categorical_features, f_cat, indent=2)
        json.dump(numerical_features, f_num, indent=2)

    # Optional console output for debug
    print("Categorical features:", len(categorical_features))
    print(categorical_features)
    print("\nNumerical features:", len(numerical_features))
    print(numerical_features)
