import kfp
from kfp import dsl

@dsl.component(
    base_image="microwave1005/scipy-img:latest"
)
def dataloader_component(
    test_output: dsl.Output[dsl.Dataset],
    train_output: dsl.Output[dsl.Dataset],

    endpoint_url: str = "http://minio.minio.svc.cluster.local:9000",
    bucket_name: str  = "sample-data",
    test_key: str     = "data/application_test.csv",
    train_key: str    = "data/application_train.csv",
    aws_access_key_id: str = "minio",
    aws_secret_access_key: str = "minio123",
):
    # All non-KFP imports are inside the function
    import boto3
    import pandas as pd
    import io

    # Connect to MinIO / S3-compatible storage
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Download and save test set
    obj_test = s3.get_object(Bucket=bucket_name, Key=test_key)
    test_df = pd.read_csv(io.BytesIO(obj_test["Body"].read()))
    test_df.to_csv(test_output.path, index=False)

    # Download and save train set
    obj_train = s3.get_object(Bucket=bucket_name, Key=train_key)
    train_df = pd.read_csv(io.BytesIO(obj_train["Body"].read()))
    train_df.to_csv(train_output.path, index=False)
