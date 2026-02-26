from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import ibis
import pyarrow as pa
import pytest
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import bigquery

import ibis_stream.bigquery as mod

pytestmark = [pytest.mark.integration, pytest.mark.bigquery]


def _env_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _LiveConfig:
    project: str
    dataset: str
    location: str | None


@pytest.fixture(scope="session")
def live_config() -> _LiveConfig:
    if not _env_truthy(os.getenv("BQ_LIVE_TESTS")):
        pytest.skip("Set BQ_LIVE_TESTS=1 to run live BigQuery integration tests.")

    project = os.getenv("BQ_PROJECT")
    dataset = os.getenv("BQ_DATASET")
    location = os.getenv("BQ_LOCATION")
    if not project or not dataset:
        pytest.fail("BQ_LIVE_TESTS=1 requires BQ_PROJECT and BQ_DATASET.")
    return _LiveConfig(project=project, dataset=dataset, location=location)


@pytest.fixture(scope="session")
def bq_client(live_config: _LiveConfig) -> bigquery.Client:
    client = bigquery.Client(project=live_config.project, location=live_config.location)
    dataset_ref = f"{live_config.project}.{live_config.dataset}"
    try:
        client.get_dataset(dataset_ref)
    except NotFound as exc:
        pytest.fail(f"Configured dataset {dataset_ref!r} does not exist or is inaccessible: {exc}")
    return client


@pytest.fixture(scope="session")
def ibis_backend(
    live_config: _LiveConfig, bq_client: bigquery.Client
) -> ibis.backends.bigquery.Backend:
    return ibis.bigquery.connect(
        project_id=live_config.project,
        dataset_id=live_config.dataset,
        client=bq_client,
        location=live_config.location,
    )


@pytest.fixture
def temp_table_factory(live_config: _LiveConfig, bq_client: bigquery.Client) -> Any:
    created: list[str] = []

    def _create(schema: list[bigquery.SchemaField]) -> tuple[str, str]:
        table_name = f"ibis_storage_it_{uuid.uuid4().hex[:18]}"
        table_path = f"{live_config.project}.{live_config.dataset}.{table_name}"
        table = bigquery.Table(table_path, schema=schema)
        bq_client.create_table(table)
        created.append(table_path)
        return table_name, table_path

    try:
        yield _create
    finally:
        for table_path in created:
            bq_client.delete_table(table_path, not_found_ok=True)


@contextmanager
def _patched_insert(**kwargs: Any) -> Any:
    with mod.stream_mode(**kwargs):
        yield


def _fetch_rows(bq_client: bigquery.Client, table_path: str) -> list[tuple[int, str | None]]:
    query = f"SELECT id, payload FROM `{table_path}` ORDER BY id"
    rows = bq_client.query(query).result()
    return [(int(row["id"]), row["payload"]) for row in rows]


def _wait_for_rows(
    bq_client: bigquery.Client,
    table_path: str,
    *,
    expected: list[tuple[int, str | None]],
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_rows: list[tuple[int, str | None]] = []
    while time.monotonic() < deadline:
        last_rows = _fetch_rows(bq_client, table_path)
        if last_rows == expected:
            return
        time.sleep(1.0)
    assert last_rows == expected


def _wait_for_aggregates(
    bq_client: bigquery.Client,
    table_path: str,
    *,
    expected_count: int,
    expected_sum: int,
    timeout_seconds: float = 30.0,
) -> None:
    query = f"SELECT COUNT(*) AS c, SUM(id) AS s FROM `{table_path}`"
    deadline = time.monotonic() + timeout_seconds
    last = (-1, -1)
    while time.monotonic() < deadline:
        row = next(iter(bq_client.query(query).result()))
        count = int(row["c"])
        total = int(row["s"] or 0)
        last = (count, total)
        if count == expected_count and total == expected_sum:
            return
        time.sleep(1.0)
    assert last == (expected_count, expected_sum)


def _seed_rows_via_query(
    bq_client: bigquery.Client,
    table_path: str,
    rows: list[tuple[int, str | None]],
) -> None:
    value_placeholders: list[str] = []
    params: list[bigquery.ScalarQueryParameter] = []
    for index, (row_id, payload) in enumerate(rows):
        id_param = f"id_{index}"
        payload_param = f"payload_{index}"
        value_placeholders.append(f"(@{id_param}, @{payload_param})")
        params.append(bigquery.ScalarQueryParameter(id_param, "INT64", row_id))
        params.append(bigquery.ScalarQueryParameter(payload_param, "STRING", payload))

    query = (
        f"INSERT INTO `{table_path}` (id, payload) VALUES " f"{', '.join(value_placeholders)}"
    )
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    bq_client.query(query, job_config=job_config).result()


def test_live_pandas_append_round_trip(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    pandas = pytest.importorskip("pandas")
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    dataframe = pandas.DataFrame(
        {
            "id": [1, 2, 3],
            "payload": ["alpha", "beta", "gamma"],
        }
    )

    with _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000):
        ibis_backend.insert(table_name, dataframe)

    _wait_for_rows(
        bq_client,
        table_path,
        expected=[(1, "alpha"), (2, "beta"), (3, "gamma")],
    )


def test_live_overwrite_replaces_existing_rows(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    pandas = pytest.importorskip("pandas")
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    _seed_rows_via_query(
        bq_client,
        table_path,
        [
            (1, "old-a"),
            (2, "old-b"),
        ],
    )

    dataframe = pandas.DataFrame(
        {
            "id": [10, 11],
            "payload": ["new-a", "new-b"],
        }
    )
    with _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000):
        ibis_backend.insert(table_name, dataframe, overwrite=True)

    _wait_for_rows(
        bq_client,
        table_path,
        expected=[(10, "new-a"), (11, "new-b")],
    )


def test_live_byte_split_stress_preserves_all_rows_once(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    pandas = pytest.importorskip("pandas")
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    row_count = 1_000
    payload = "x" * 256
    dataframe = pandas.DataFrame(
        {
            "id": list(range(row_count)),
            "payload": [payload] * row_count,
        }
    )

    with _patched_insert(target_batch_bytes=16 * 1024, max_batch_rows=50_000):
        ibis_backend.insert(table_name, dataframe)

    expected_sum = row_count * (row_count - 1) // 2
    _wait_for_aggregates(
        bq_client,
        table_path,
        expected_count=row_count,
        expected_sum=expected_sum,
    )


def test_live_invalid_rows_raise_insert_error(
    ibis_backend: ibis.backends.bigquery.Backend,
    temp_table_factory: Any,
) -> None:
    pandas = pytest.importorskip("pandas")
    table_name, _table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    dataframe = pandas.DataFrame(
        {
            "id": [1, None],
            "payload": ["ok", "bad-null-required-id"],
        }
    )

    with (
        _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000),
        pytest.raises((mod.StorageWriteInsertError, GoogleAPICallError)) as caught,
    ):
        ibis_backend.insert(table_name, dataframe)

    if isinstance(caught.value, mod.StorageWriteInsertError):
        assert "row" in str(caught.value).lower()


def test_live_record_batch_reader_insert_round_trip(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("payload", pa.string()),
        ]
    )
    batches = [
        pa.record_batch([pa.array([1, 2]), pa.array(["a", "b"])], schema=schema),
        pa.record_batch([pa.array([3]), pa.array(["c"])], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    with _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000):
        ibis_backend.insert(table_name, reader)

    _wait_for_rows(
        bq_client,
        table_path,
        expected=[(1, "a"), (2, "b"), (3, "c")],
    )


def test_live_list_dict_insert_round_trip(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    rows = [
        {"id": 1, "payload": "a"},
        {"id": 2, "payload": "b"},
        {"id": 3, "payload": "c"},
    ]

    with _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000):
        ibis_backend.insert(table_name, rows)

    _wait_for_rows(
        bq_client,
        table_path,
        expected=[(1, "a"), (2, "b"), (3, "c")],
    )


def test_live_pandas_upsert_round_trip(
    ibis_backend: ibis.backends.bigquery.Backend,
    bq_client: bigquery.Client,
    temp_table_factory: Any,
) -> None:
    pandas = pytest.importorskip("pandas")
    table_name, table_path = temp_table_factory(
        [
            bigquery.SchemaField("id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("payload", "STRING"),
        ]
    )
    _seed_rows_via_query(
        bq_client,
        table_path,
        [
            (1, "old-a"),
            (2, "old-b"),
        ],
    )

    dataframe = pandas.DataFrame(
        {
            "id": [2, 3],
            "payload": ["new-b", "new-c"],
        }
    )
    with _patched_insert(target_batch_bytes=256 * 1024, max_batch_rows=10_000):
        ibis_backend.upsert(table_name, dataframe, "id")

    _wait_for_rows(
        bq_client,
        table_path,
        expected=[(1, "old-a"), (2, "new-b"), (3, "new-c")],
    )
