# ibis-stream

[![Checks](https://github.com/unlap-hq/ibis-stream/actions/workflows/checks.yml/badge.svg)](https://github.com/unlap-hq/ibis-stream/actions/workflows/checks.yml)
[![PyPI version](https://img.shields.io/pypi/v/ibis-stream.svg)](https://pypi.org/project/ibis-stream/)

`ibis-stream` provides streaming insert helpers for [ibis](https://github.com/ibis-project/ibis).

Right now, the project focuses on **BigQuery**: inside a context manager, it patches Ibis BigQuery `insert`/`upsert` so in-memory data is written through the BigQuery Storage Write API.

## Why this exists

For in-memory payloads, Storage Write can be a better fit for low-latency incremental writes than the default insert path.

## Current scope

- BigQuery support via `ibis_stream.bigquery.stream_mode`
- `insert` support, including `overwrite=True`
- `upsert` support via a temporary staging table + merge
- Automatic fallback to the original Ibis behavior when data is not an in-memory payload

## Snowflake (coming soon)

Planned support will target Snowflake streaming ingestion via **Snowpipe Streaming**.

Expected implementation shape:

- `insert`: coerce in-memory data to Arrow-compatible batches and stream them through Snowpipe Streaming channels.
- `upsert`: stream into a staging table, then execute `MERGE` into the target table on the configured key.
- `overwrite=True`: truncate target table before ingesting the replacement batch set.

Likely requirements:

- Snowflake account and role permissions for streaming ingest and target-table writes.
- A backend-specific configuration surface for connection/auth settings.

This is similar in goal to BigQuery Storage Write (low-latency ingest), but the SDK/API surface and operational model are Snowflake-specific.

## Redshift (coming soon)

Planned support will use Redshift-native loading patterns for in-memory data.

Expected implementation shape:

- `insert`: convert input to columnar micro-batches (for example Parquet), write to a temporary S3 prefix, then load with `COPY`.
- `upsert`: `COPY` into a staging table, then `MERGE` into the target table on the configured key.
- `overwrite=True`: truncate target table before running the replacement load.

Likely requirements:

- Configured temporary S3 bucket/prefix for batch files.
- IAM role/policy and Redshift permissions needed for `COPY`, `MERGE`, and staging-table lifecycle operations.

Redshift also has Streaming Ingestion for Kinesis/MSK event streams, but for application-driven DataFrame/Arrow writes, the `COPY`-based flow is typically the practical fit.

## Installation

Install with pip:

```bash
python -m pip install "ibis-stream[bigquery]"
```

For development dependencies:

```bash
python -m pip install --group dev ".[bigquery]"
```

## Quick start

```python
import ibis
import pandas as pd
from google.cloud import bigquery

from ibis_stream.bigquery import stream_mode

client = bigquery.Client(project="my-project", location="US")
con = ibis.bigquery.connect(
    project_id="my-project",
    dataset_id="my_dataset",
    client=client,
    location="US",
)

rows = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "payload": ["a", "b", "c"],
    }
)

with stream_mode():
    con.insert("target_table", rows)
```

Upsert example:

```python
with stream_mode():
    con.upsert("target_table", rows, "id")
```

## API

### `stream_mode(...)`

Context manager that installs BigQuery insert/upsert hooks for the duration of the `with` block.

```python
stream_mode(
    target_batch_bytes=8 * 1024 * 1024,
    max_batch_rows=50_000,
    append_timeout_seconds=120.0,
)
```

- `target_batch_bytes`: target serialized Arrow batch size. Must be `> 0` and `<= 9 * 1024 * 1024`.
- `max_batch_rows`: maximum rows per emitted batch chunk.
- `append_timeout_seconds`: timeout for each append request (`None` disables timeout).

### `StorageWriteInsertError`

Raised for Storage Write failures surfaced by this patch layer (for example, row-level append errors).

## Supported in-memory inputs

- `pyarrow.Table`
- `pyarrow.RecordBatch`
- `pyarrow.RecordBatchReader`
- pandas DataFrame
- polars DataFrame
- `list[dict]` and similar objects that Ibis can coerce to `memtable`
- `ibis.memtable(...)` expressions

Non-in-memory table expressions fall back to the original backend `insert`/`upsert` implementation.

## Development with Pixi

```bash
pixi run lint
pixi run format-check
pixi run test-unit
pixi run test-integration-bigquery
```

Run unit tests:

```bash
pytest -q tests/unit
```

Run integration tests:

```bash
python scripts/integration.py run --dialect bigquery
```

Enable live BigQuery integration tests:

```bash
export BQ_LIVE_TESTS=1
export BQ_PROJECT=<your-gcp-project>
export BQ_DATASET=<existing-dataset>
export BQ_LOCATION=US  # optional

pytest -q tests/integration/bigquery/test_live.py -m "integration and bigquery"
```

## License

Apache-2.0. See [LICENSE](LICENSE).
