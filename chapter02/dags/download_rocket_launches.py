import json
import pathlib
from datetime import datetime, timedelta

import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="download_rocket_launches",
    description="Download rocket pictures of recently launched rockets.",
    start_date=datetime.now() - timedelta(days=14),
    schedule="@daily",
)

download_launches = BashOperator(
    task_id="download_launches",
    bash_command="curl -f -o /opt/airflow/launches.json -L 'https://ll.thespacedevs.com/2.0.0/launch/upcoming' || echo 'Failed to download launches'",
    dag=dag,
)


def _get_pictures():
    # Ensure directory exists
    pathlib.Path("/opt/airflow/images").mkdir(parents=True, exist_ok=True)

    # Download all pictures in launches.json
    with open("/opt/airflow/launches.json") as f:
        launches = json.load(f)
        image_urls = [launch["image"] for launch in launches["results"]]
        for image_url in image_urls:
            try:
                response = requests.get(image_url)
                image_filename = image_url.split("/")[-1]
                target_file = f"/opt/airflow/images/{image_filename}"
                with open(target_file, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {image_url} to {target_file}")
            except requests_exceptions.MissingSchema:
                print(f"{image_url} appears to be an invalid URL.")
            except requests_exceptions.ConnectionError:
                print(f"Could not connect to {image_url}.")


get_pictures = PythonOperator(
    task_id="get_pictures", python_callable=_get_pictures, dag=dag
)

notify = BashOperator(
    task_id="notify",
    bash_command='echo "There are now $(ls /opt/airflow/images/ | wc -l) images."',
    dag=dag,
)

download_launches >> get_pictures >> notify
