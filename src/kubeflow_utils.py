import requests
from pathlib import Path
import yaml
from loguru import logger

CONFIG_PATH = (Path(__file__).parent.parent / "config.yaml").resolve()

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

USERNAME = config['kubeflow']['username']
PASSWORD = config['kubeflow']['password']
NAMESPACE = config['kubeflow']['namespace']
HOST = config['kubeflow']['host']


def get_session_cookie():
    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    return session_cookie


def get_or_create_pipeline(
    client,
    pipeline_name: str,
    pipeline_package_path: str,
    version: str,
    pipeline_description: str,
):
    pipeline_id = client.get_pipeline_id(pipeline_name)

    if pipeline_id is None:
        logger.info(f"Creating a new pipeline: {pipeline_name}")
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_package_path,
            pipeline_name=pipeline_name,
            description=pipeline_description,
        )
    else:
        logger.info(f"Retrieving the existing pipeline: {pipeline_name}")
        pipeline = client.get_pipeline(pipeline_id)

    pipeline_version = client.upload_pipeline_version(
        pipeline_package_path=pipeline_package_path,
        pipeline_version_name=f"{pipeline_name} {version}",
        pipeline_id=pipeline_id,
    )

    return pipeline_version


def get_or_create_experiment(client, name: str, namespace: str):
    try:
        experiment = client.get_experiment(experiment_name=name, namespace=namespace)
    except Exception:
        logger.info(f"Creating new experiment: {name}")
        experiment = client.create_experiment(experiment_name=name, namespace=namespace)

    return experiment
