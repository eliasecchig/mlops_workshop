
import argparse

import kfp.v2.compiler as compiler
import kfp.v2.dsl as dsl


@dsl.component(packages_to_install=["scikit-learn", "pandas", "joblib"])
def model_training_op(
        dataset: dsl.Input[dsl.Dataset],
        model: dsl.Output[dsl.Model]
):
    import glob
    import json
    import os

    import joblib
    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import train_test_split

    TARGET_COLUMN = "tip_bin"
    TARGET_LABELS = ["tip<20%", "tip>=20%"]

    def sanitize(path: str) -> str:
        return path.replace("gs://", "/gcs/", 1) if path and path.startswith("gs://") else path

    def get_dataframe(path: str):
        if os.path.isdir(path):  # base data directory is passed
            files = glob.glob(f"{path}/*.csv")
        elif "*" in path:  # a glob expression is passed
            files = glob.glob(path)
        else:  # single file is passed
            files = [path]
        dfs = (pd.read_csv(f, header=0) for f in files)
        return pd.concat(dfs, ignore_index=True)

    def create_datasets(training_data_dir: str, validation_data_dir: str):
        """Creates training and validation datasets."""

        train_dataset = get_dataframe(training_data_dir)

        if validation_data_dir:
            return train_dataset, get_dataframe(validation_data_dir)
        else:
            return train_test_split(train_dataset, test_size=.25, random_state=42)

    def log_metrics(y_pred: pd.Series, y_true: pd.Series, output_dir: str):
        curve = roc_curve(y_score=y_pred, y_true=y_true)
        auc = roc_auc_score(y_score=y_pred, y_true=y_true)
        cm = confusion_matrix(labels=[False, True], y_pred=y_pred, y_true=y_true)

        with open(f"{output_dir}/metrics.json", "w") as f:
            metrics = {"auc": auc}
            metrics["confusion_matrix"] = {}
            metrics["confusion_matrix"]["categories"] = TARGET_LABELS
            metrics["confusion_matrix"]["matrix"] = cm.tolist()
            metrics["roc_curve"] = {}
            metrics["roc_curve"]["fpr"] = curve[0].tolist()
            metrics["roc_curve"]["tpr"] = curve[1].tolist()
            metrics["roc_curve"]["thresholds"] = curve[2].tolist()
            json.dump(metrics, f, indent=2)

    def split(df: pd.DataFrame):
        return df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]

    def train(training_data_dir: str, validation_data_dir: str, output_dir: str):
        train_df, val_df = create_datasets(training_data_dir, validation_data_dir)

        X_train, y_train = split(train_df)
        X_test, y_test = split(val_df)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, f"{output_dir}/model.joblib")

        y_pred = model.predict(X_test)
        log_metrics(y_pred, y_test, output_dir)

        return model.score(X_test, y_test)

    train(f"{dataset.path}/train", f"{dataset.path}/val", f"{model.path}")


@dsl.component()
def data_validation_op(dataset: dsl.Input[dsl.Dataset]) -> str:
    return "valid"


@dsl.component()
def data_preparation_op():
    pass


@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def model_validation_op(
        metrics: dsl.Input[dsl.ClassificationMetrics],
        threshold_auc: float = 0.50
) -> str:
    return "valid" if metrics.metadata["auc"] > threshold_auc else 'not_valid'


@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def model_upload_op(
        model: dsl.Input[dsl.Model],
        serving_container_image_uri: str,
        project_id: str,
        location: str,
        model_name: str) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    parent_model = matches[0].resource_name if matches else None

    registered_model = aiplatform.Model.upload(
        display_name=model_name,
        parent_model=parent_model,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_container_image_uri
    )

    return registered_model.versioned_resource_name


@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def model_evaluation_upload_op(
        metrics: dsl.Input[dsl.ClassificationMetrics],
        model_resource_name: str,
        project_id: str,
        location: str):
    from google.api_core import gapic_v1
    from google.cloud import aiplatform
    from google.protobuf.struct_pb2 import Struct
    from google.protobuf.struct_pb2 import Value

    model_evaluation = {
        "display_name": "pipeline-eval",
        "metrics": Value(struct_value=Struct(fields={"auRoc": Value(number_value=metrics.metadata["auc"])})),
        "metrics_schema_uri": "gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml"
    }

    aiplatform.init(project=project_id, location=location)
    api_endpoint = location + '-aiplatform.googleapis.com'
    client = aiplatform.gapic.ModelServiceClient(client_info=gapic_v1.client_info.ClientInfo(
        user_agent="google-cloud-pipeline-components"),
        client_options={
            "api_endpoint": api_endpoint,
        })
    client.import_model_evaluation(parent=model_resource_name, model_evaluation=model_evaluation)


@dsl.component()
def model_evaluation_op(model: dsl.Input[dsl.Model], metrics: dsl.Output[dsl.ClassificationMetrics]):
    import json

    with open(f"{model.path}/metrics.json", "r") as f:
        model_metrics = json.load(f)

    conf_matrix = model_metrics["confusion_matrix"]
    metrics.log_confusion_matrix(categories=conf_matrix["categories"], matrix=conf_matrix["matrix"])

    curve = model_metrics["roc_curve"]
    metrics.log_roc_curve(fpr=curve["fpr"], tpr=curve["tpr"], threshold=curve["thresholds"])

    metrics.metadata["auc"] = model_metrics["auc"]


@dsl.component(packages_to_install=["google-cloud-bigquery"])
def data_extract_op(project_id: str, location: str, dataset: dsl.Output[dsl.Dataset]):
    import os

    from google.cloud import bigquery

    client = bigquery.Client()
    query = """
    EXPORT DATA OPTIONS(
        uri='{path}/*.csv',
        format='CSV',
        overwrite=true,
        header=true,
        field_delimiter=',') AS
    SELECT
        EXTRACT(MONTH from pickup_datetime) as trip_month,
        EXTRACT(DAY from pickup_datetime) as trip_day,
        EXTRACT(DAYOFWEEK from pickup_datetime) as trip_day_of_week,
        EXTRACT(HOUR from pickup_datetime) as trip_hour,
        TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) as trip_duration,
        trip_distance,
        payment_type,
        pickup_location_id as pickup_zone,
        pickup_location_id as dropoff_zone,
        IF((SAFE_DIVIDE(tip_amount, fare_amount) >= 0.2), 1, 0) AS tip_bin
    FROM
        `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_{year}` TABLESAMPLE SYSTEM (1 PERCENT)
    WHERE
        TIMESTAMP_DIFF(dropoff_datetime, pickup_datetime, SECOND) BETWEEN 300 AND 10800
    LIMIT {limit}
    """
    datasets = [
        (f"{dataset.path}/train", 2020, 10000),
        (f"{dataset.path}/val", 2020, 5000),
        (f"{dataset.path}/test", 2020, 1000)
    ]
    for ds in datasets:
        path = ds[0].replace("/gcs/", "gs://", 1)
        os.makedirs(path, exist_ok=True)
        # ignoring the provided location as this dataset is in US
        job = client.query(query.format(path=path, year=ds[1], limit=ds[2]), project=project_id, location="us")
        job.result()



@dsl.pipeline(name="taxi-tips-training")
def training_pipeline(
        project_id: str, location: str):
    model_name = "taxi-tips"

    data_extraction_task = data_extract_op(
        project_id=project_id, location=location
    ).set_display_name("extract-data")

    data_validation_task = data_validation_op(
        dataset=data_extraction_task.outputs["dataset"]
    ).set_display_name("validate-data")

    data_preparation_task = data_preparation_op().set_display_name("prepare-data")
    data_preparation_task.after(data_validation_task)

    training_task = model_training_op(
        dataset=data_extraction_task.outputs["dataset"],
    ).set_display_name("train-model")
    training_task.after(data_preparation_task)

    model_evaluation_task = model_evaluation_op(
        model=training_task.outputs["model"]
    ).set_display_name("evaluate-model")

    model_validation_task = model_validation_op(
        metrics=model_evaluation_task.outputs["metrics"],
    ).set_display_name("validate-model")

    with dsl.Condition(model_validation_task.output == "valid", name="check-performance"):
        model_upload_task = model_upload_op(
            model=training_task.outputs["model"],
            model_name=model_name,
            serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest",
            project_id=project_id,
            location=location
        ).set_display_name("register-model")

        model_evaluation_upload_task = model_evaluation_upload_op(
            metrics=model_evaluation_task.outputs["metrics"],
            model_resource_name=model_upload_task.output,
            project_id=project_id,
            location=location
        ).set_display_name("register-model-evaluation")

def compile(filename: str):
    cmp = compiler.Compiler()
    cmp.compile(pipeline_func=training_pipeline, package_path=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-file-name", type=str, default="pipeline.json")

    args = parser.parse_args()

    compile(args.pipeline_file_name)
