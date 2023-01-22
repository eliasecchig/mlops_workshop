import os

from google.cloud import aiplatform
import google.auth

PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    creds, PROJECT_ID = google.auth.default()

REGION = os.environ["REGION"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
PIPELINE_NAME = os.environ["PIPELINE_NAME"]
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "dummy-experiment")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME","dummy-endpoint")

aiplatform.init(project=PROJECT_ID, location=REGION)
sync_pipeline = os.getenv("SUBMIT_PIPELINE_SYNC", 'False').lower() in ('true', '1', 't')

job = aiplatform.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path='pipeline.json',
    location=REGION,
    project=PROJECT_ID,
    enable_caching=True,
    pipeline_root=f'gs://{BUCKET_NAME}',
    parameter_values={'project_id':PROJECT_ID, 'location':REGION}
)
print(f"Submitting pipeline {PIPELINE_NAME} in experiment {EXPERIMENT_NAME}.")
job.submit(experiment=EXPERIMENT_NAME)

if sync_pipeline:
    job.wait()
