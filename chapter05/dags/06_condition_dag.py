import airflow
import pendulum

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

from datetime import datetime, timedelta

ERP_CHANGE_DATE = pendulum.now("UTC") - timedelta(days=1)


def _pick_erp_system(**context):
    if context["logical_date"] < ERP_CHANGE_DATE:
        return "fetch_sales_old"
    else:
        return "fetch_sales_new"


def _latest_only(**context):
    """
    この関数は、与えられたデータインターバルが、現在時刻と比較して
    特定の時間枠内にある場合にのみタスクの実行を許可し、そうでない場合はスキップします。

    主な変更点:
    - `context["execution_date"]` の代わりに `context["data_interval_end"]` を使用。
    - `pendulum.now("UTC")` を直接使用して現在時刻を取得。
    - `dag.following_schedule` の代わりに、`data_interval_end` と現在時刻の差分で判断。
      これは `LatestOnlyOperator` の直接的な代替ではなく、
      カスタムの時間ベースのスキップロジックの例です。
      もし「最新のDAG Runのみを実行する」ことが目的であれば、
      `airflow.operators.latest_only.LatestOnlyOperator` を使用する方が推奨されます。
    """
    now_utc = pendulum.now("UTC")
    current_data_interval_end = context["data_interval_end"]

    # 例: current_data_interval_end が現在時刻から1日以上前であればスキップする
    # この条件は、あなたの「最新」の定義に合わせて調整してください。
    # 例えば、特定のスケジュール期間外のものをスキップしたい場合など。
    time_difference = now_utc - current_data_interval_end

    # この例では、data_interval_end が現在時刻より未来であるか、
    # または過去1日以内のものでなければスキップします。
    # これは、非常に古い、または将来の意図しない実行を防ぐのに役立ちます。
    if time_difference > pendulum.duration(days=1) or current_data_interval_end > now_utc:
        print(f"Skipping task: data_interval_end {current_data_interval_end} is out of the acceptable window (now: {now_utc}).")
        raise AirflowSkipException()
    else:
        print(f"Proceeding: data_interval_end {current_data_interval_end} is within the acceptable window (now: {now_utc}).")



with DAG(
    dag_id="06_condition_dag",
    start_date=datetime.now() - timedelta(days=3),
    schedule="@daily",
) as dag:
    start = EmptyOperator(task_id="start")

    pick_erp = BranchPythonOperator(
        task_id="pick_erp_system", python_callable=_pick_erp_system
    )

    fetch_sales_old = EmptyOperator(task_id="fetch_sales_old")
    clean_sales_old = EmptyOperator(task_id="clean_sales_old")

    fetch_sales_new = EmptyOperator(task_id="fetch_sales_new")
    clean_sales_new = EmptyOperator(task_id="clean_sales_new")

    join_erp = EmptyOperator(task_id="join_erp_branch", trigger_rule="none_failed")

    fetch_weather = EmptyOperator(task_id="fetch_weather")
    clean_weather = EmptyOperator(task_id="clean_weather")

    join_datasets = EmptyOperator(task_id="join_datasets")
    train_model = EmptyOperator(task_id="train_model")

    latest_only = PythonOperator(task_id="latest_only", python_callable=_latest_only)

    deploy_model = EmptyOperator(task_id="deploy_model")

    start >> [pick_erp, fetch_weather]
    pick_erp >> [fetch_sales_old, fetch_sales_new]
    fetch_sales_old >> clean_sales_old
    fetch_sales_new >> clean_sales_new
    [clean_sales_old, clean_sales_new] >> join_erp
    fetch_weather >> clean_weather
    [join_erp, clean_weather] >> join_datasets
    join_datasets >> train_model >> deploy_model
    latest_only >> deploy_model
