from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import chain
from typing import Any

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
import pyarrow as pa
import sqlglot as sg
from google.cloud.bigquery_storage_v1 import BigQueryWriteClient
from google.cloud.bigquery_storage_v1.types import (
    AppendRowsRequest,
    AppendRowsResponse,
    ArrowRecordBatch,
    ArrowSchema,
)
from google.cloud.bigquery_storage_v1.writer import AppendRowsStream

_PATCH_LOCK = threading.Lock()
_WRITE_CLIENT_ATTR = "_ibis_stream_bigquery_write_client"
_DEFAULT_TARGET_BATCH_BYTES = 8 * 1024 * 1024
# BigQuery AppendRows has a 10 MB request limit. Keep headroom for protobuf framing.
_MAX_TARGET_BATCH_BYTES = 9 * 1024 * 1024
_ArrowInput = pa.Table | pa.RecordBatch | pa.RecordBatchReader


@dataclass(frozen=True)
class StorageWritePatchOptions:
    """Configuration for BigQuery inserts routed through Storage Write API."""

    target_batch_bytes: int = _DEFAULT_TARGET_BATCH_BYTES
    max_batch_rows: int = 50_000
    append_timeout_seconds: float | None = 120.0

    def __post_init__(self) -> None:
        if self.target_batch_bytes <= 0:
            raise ValueError("target_batch_bytes must be greater than zero")
        if self.target_batch_bytes > _MAX_TARGET_BATCH_BYTES:
            raise ValueError(
                f"target_batch_bytes must be <= {_MAX_TARGET_BATCH_BYTES} bytes "
                "(BigQuery AppendRows requests must remain under 10 MB)"
            )
        if self.max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be greater than zero")
        if self.append_timeout_seconds is not None and self.append_timeout_seconds <= 0:
            raise ValueError("append_timeout_seconds must be greater than zero when provided")


@dataclass(frozen=True)
class _ResolvedBigQueryTable:
    project: str
    dataset: str
    table: str


class StorageWriteInsertError(RuntimeError):
    """Raised when BigQuery Storage Write API append fails."""


@dataclass
class _PatchState:
    original_insert: Callable[..., Any] | None = None
    original_upsert: Callable[..., Any] | None = None
    patched: bool = False
    active_contexts: int = 0


_STATE = _PatchState()
_ACTIVE_OPTIONS: ContextVar[StorageWritePatchOptions | None] = ContextVar(
    "ibis_stream_bigquery_active_options",
    default=None,
)


def _coerce_to_arrow_input(obj: Any) -> _ArrowInput | None:
    if isinstance(obj, (pa.Table, pa.RecordBatch, pa.RecordBatchReader)):
        return obj
    return None


def _coerce_to_table_expr(obj: Any) -> ir.Table:
    if isinstance(obj, ir.Table):
        return obj
    return ibis.memtable(obj)


def _in_memory_table_to_arrow_input(table_expr: ir.Table) -> pa.Table | None:
    op = table_expr.op()
    if not isinstance(op, ops.InMemoryTable):
        return None
    return op.data.to_pyarrow(op.schema)


def _resolve_table(
    backend: Any,
    *,
    name: str,
    database: str | tuple[str, str] | None,
) -> _ResolvedBigQueryTable:
    def _normalize_name_part(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _part_name(part: Any, *, dialect: str) -> str | None:
        if part is None:
            return None
        if isinstance(part, str):
            return _normalize_name_part(part)
        if hasattr(part, "name"):
            return _normalize_name_part(part.name)
        return _normalize_name_part(part.sql(dialect=dialect))

    def _normalize_dataset_for_project(project: str | None, dataset: str | None) -> str | None:
        if dataset is None:
            return None
        if "." not in dataset:
            return dataset
        if project and dataset.startswith(f"{project}."):
            return dataset[len(project) + 1 :]
        raise ValueError(f"Resolved invalid BigQuery dataset identifier {dataset!r}")

    table_loc = backend._to_sqlglot_table(database)
    catalog, db = backend._to_catalog_db_tuple(table_loc)
    if catalog is None:
        catalog = backend.current_catalog
    if db is None:
        db = backend.current_database

    dialect = getattr(backend, "dialect", "bigquery")
    try:
        table_name_expr = sg.to_table(name, dialect=dialect)
    except Exception as exc:
        raise ValueError(
            f"Unable to resolve BigQuery table path from name {name!r} and database {database!r}"
        ) from exc

    name_parts = [_part_name(part, dialect=dialect) for part in table_name_expr.parts]
    if any(part is None for part in name_parts):
        raise ValueError(
            f"Unable to resolve BigQuery table path from name {name!r} and database {database!r}"
        )

    default_project = _normalize_name_part(catalog)
    default_dataset = _normalize_name_part(db)
    default_dataset = _normalize_dataset_for_project(default_project, default_dataset)

    if len(name_parts) == 1:
        project, dataset, table = default_project, default_dataset, name_parts[0]
    elif len(name_parts) == 2:
        project, dataset, table = default_project, name_parts[0], name_parts[1]
    elif len(name_parts) == 3:
        project, dataset, table = name_parts
    else:
        raise ValueError(
            f"Unable to resolve BigQuery table path from name {name!r} and database {database!r}"
        )

    dataset = _normalize_dataset_for_project(project, dataset)

    if not project or not dataset or not table:
        raise ValueError(
            f"Resolved invalid BigQuery target project={project!r}, dataset={dataset!r}, table={table!r}"
        )
    return _ResolvedBigQueryTable(project=project, dataset=dataset, table=table)


def _build_append_request_for_batch(
    batch: pa.RecordBatch,
    *,
    serialized_batch: bytes,
) -> AppendRowsRequest:
    return AppendRowsRequest(
        arrow_rows=AppendRowsRequest.ArrowData(
            rows=ArrowRecordBatch(
                serialized_record_batch=serialized_batch,
                row_count=batch.num_rows,
            )
        )
    )


def _format_row_error_summary(response: AppendRowsResponse) -> str:
    if not response.row_errors:
        return "unknown row error"
    first = response.row_errors[0]
    index = getattr(first, "index", None)
    message = getattr(first, "message", "row append failed")
    return f"first_row_index={index}: {message}"


def _raise_row_errors(response: AppendRowsResponse) -> None:
    if not response.row_errors:
        return
    raise StorageWriteInsertError(
        f"BigQuery Storage Write API returned row errors: {_format_row_error_summary(response)}"
    )


def _get_or_create_write_client(backend: Any) -> BigQueryWriteClient:
    cached = getattr(backend, _WRITE_CLIENT_ATTR, None)
    if cached is not None:
        return cached

    credentials = getattr(getattr(backend, "client", None), "_credentials", None)
    client = BigQueryWriteClient(credentials=credentials)
    setattr(backend, _WRITE_CLIENT_ATTR, client)
    return client


def _serialize_record_batch(batch: pa.RecordBatch) -> bytes:
    return batch.serialize().to_pybytes()


def _iter_row_chunks(batch: pa.RecordBatch, *, max_batch_rows: int) -> Iterator[pa.RecordBatch]:
    for offset in range(0, batch.num_rows, max_batch_rows):
        yield batch.slice(offset, max_batch_rows)


def _iter_source_batches(
    arrow_input: _ArrowInput,
    *,
    max_batch_rows: int,
) -> Iterator[pa.RecordBatch]:
    if isinstance(arrow_input, pa.Table):
        yield from arrow_input.to_batches(max_chunksize=max_batch_rows)
        return

    if isinstance(arrow_input, pa.RecordBatch):
        yield from _iter_row_chunks(arrow_input, max_batch_rows=max_batch_rows)
        return

    for batch in arrow_input:
        if batch.num_rows == 0:
            continue
        yield from _iter_row_chunks(batch, max_batch_rows=max_batch_rows)


def _iter_storage_write_batches(
    arrow_input: _ArrowInput,
    *,
    options: StorageWritePatchOptions,
) -> Iterator[tuple[pa.RecordBatch, bytes]]:
    for batch in _iter_source_batches(
        arrow_input,
        max_batch_rows=options.max_batch_rows,
    ):
        if batch.num_rows == 0:
            continue
        yield from _split_batch_by_target_bytes(
            batch,
            target_batch_bytes=options.target_batch_bytes,
        )


def _split_batch_by_target_bytes(
    batch: pa.RecordBatch,
    *,
    target_batch_bytes: int,
) -> Iterator[tuple[pa.RecordBatch, bytes]]:
    serialized_batch = _serialize_record_batch(batch)
    if len(serialized_batch) <= target_batch_bytes or batch.num_rows <= 1:
        yield batch, serialized_batch
        return

    midpoint = batch.num_rows // 2
    if midpoint <= 0 or midpoint >= batch.num_rows:
        yield batch, serialized_batch
        return

    yield from _split_batch_by_target_bytes(
        batch.slice(0, midpoint),
        target_batch_bytes=target_batch_bytes,
    )
    yield from _split_batch_by_target_bytes(
        batch.slice(midpoint),
        target_batch_bytes=target_batch_bytes,
    )


def _append_arrow_via_storage_write(
    backend: Any,
    *,
    target: _ResolvedBigQueryTable,
    arrow_input: _ArrowInput,
    options: StorageWritePatchOptions,
) -> None:
    batched = _iter_storage_write_batches(arrow_input, options=options)
    first_batch = next(batched, None)
    if first_batch is None:
        return

    write_client = _get_or_create_write_client(backend)
    table_path = write_client.table_path(target.project, target.dataset, target.table)
    write_stream = f"{table_path}/streams/_default"

    initial_request = AppendRowsRequest(
        write_stream=write_stream,
        arrow_rows=AppendRowsRequest.ArrowData(
            writer_schema=ArrowSchema(serialized_schema=arrow_input.schema.serialize().to_pybytes())
        ),
    )

    stream = AppendRowsStream(write_client, initial_request)
    write_error: Exception | None = None
    try:
        for batch, serialized_batch in chain((first_batch,), batched):
            request = _build_append_request_for_batch(
                batch,
                serialized_batch=serialized_batch,
            )
            response = stream.send(request).result(timeout=options.append_timeout_seconds)
            _raise_row_errors(response)
    except Exception as exc:
        write_error = exc
        raise
    finally:
        try:
            stream.close()
        except Exception as close_error:
            if write_error is None:
                raise StorageWriteInsertError(
                    "Failed to close BigQuery Storage Write stream"
                ) from close_error
            if hasattr(write_error, "add_note"):
                write_error.add_note(
                    f"Additionally failed to close BigQuery Storage Write stream: {close_error!r}"
                )


def _source_schema_from_arrow_input(arrow_input: _ArrowInput) -> ibis.Schema:
    return ibis.schema(arrow_input.schema)


def _new_staging_table_name() -> str:
    return f"_ibis_stream_upsert_{uuid.uuid4().hex[:24]}"


def _upsert_via_storage_write(
    backend: Any,
    *,
    name: str,
    on: str,
    database: str | None,
    arrow_input: _ArrowInput,
    options: StorageWritePatchOptions,
) -> None:
    original_upsert = _STATE.original_upsert
    if original_upsert is None:
        raise RuntimeError("stream_mode upsert patch has not been initialized")

    target = _resolve_table(backend, name=name, database=database)
    source_schema = _source_schema_from_arrow_input(arrow_input)
    target_schema = backend.table(name, database=database).schema()

    if on not in target_schema:
        raise ValueError(f"Upsert key {on!r} not found in target table {name!r}")
    if on not in source_schema:
        raise ValueError(f"Upsert key {on!r} not found in source data")

    staging_table = _new_staging_table_name()
    staging_database = target.dataset
    staging_namespace = (target.project, target.dataset)
    backend.create_table(staging_table, schema=source_schema, database=staging_database)

    write_or_merge_error: Exception | None = None
    try:
        _append_arrow_via_storage_write(
            backend,
            target=_ResolvedBigQueryTable(target.project, target.dataset, staging_table),
            arrow_input=arrow_input,
            options=options,
        )
        staging_expr = backend.table(staging_table, database=staging_namespace)
        original_upsert(backend, name, staging_expr, on, database=database)
    except Exception as exc:
        write_or_merge_error = exc
        raise
    finally:
        try:
            backend.drop_table(staging_table, database=staging_namespace, force=True)
        except Exception as drop_error:
            if write_or_merge_error is None:
                raise StorageWriteInsertError(
                    "Failed to drop BigQuery upsert staging table"
                ) from drop_error
            if hasattr(write_or_merge_error, "add_note"):
                write_or_merge_error.add_note(
                    f"Additionally failed to drop BigQuery upsert staging table: {drop_error!r}"
                )


def _patched_insert(
    self: Any,
    name: str,
    /,
    obj: Any,
    *,
    database: str | None = None,
    overwrite: bool = False,
) -> None:
    original = _STATE.original_insert
    if original is None:
        raise RuntimeError("stream_mode patch has not been initialized")

    options = _ACTIVE_OPTIONS.get()
    if options is None:
        original(self, name, obj, database=database, overwrite=overwrite)
        return

    arrow_input = _coerce_to_arrow_input(obj)
    if arrow_input is None:
        table_expr = _coerce_to_table_expr(obj)
        arrow_input = _in_memory_table_to_arrow_input(table_expr)
        if arrow_input is None:
            original(self, name, table_expr, database=database, overwrite=overwrite)
            return

    target = _resolve_table(self, name=name, database=database)
    if overwrite:
        self.truncate_table(name, database=(target.project, target.dataset))
    _append_arrow_via_storage_write(
        self,
        target=target,
        arrow_input=arrow_input,
        options=options,
    )


def _patched_upsert(
    self: Any,
    name: str,
    /,
    obj: Any,
    on: str,
    *,
    database: str | None = None,
) -> None:
    original = _STATE.original_upsert
    if original is None:
        raise RuntimeError("stream_mode upsert patch has not been initialized")

    options = _ACTIVE_OPTIONS.get()
    if options is None:
        original(self, name, obj, on, database=database)
        return

    arrow_input = _coerce_to_arrow_input(obj)
    if arrow_input is None:
        table_expr = _coerce_to_table_expr(obj)
        arrow_input = _in_memory_table_to_arrow_input(table_expr)
        if arrow_input is None:
            original(self, name, table_expr, on, database=database)
            return

    _upsert_via_storage_write(
        self,
        name=name,
        on=on,
        database=database,
        arrow_input=arrow_input,
        options=options,
    )


def _install_hook_locked() -> None:
    from ibis.backends.bigquery import Backend

    if _STATE.patched:
        return
    if _STATE.original_insert is None:
        _STATE.original_insert = Backend.insert
    if _STATE.original_upsert is None:
        _STATE.original_upsert = Backend.upsert
    Backend.insert = _patched_insert
    Backend.upsert = _patched_upsert
    _STATE.patched = True


def _uninstall_hook_locked() -> None:
    if not _STATE.patched:
        return
    if _STATE.original_insert is None or _STATE.original_upsert is None:
        raise RuntimeError("Missing original BigQuery insert/upsert implementation")

    from ibis.backends.bigquery import Backend

    Backend.insert = _STATE.original_insert
    Backend.upsert = _STATE.original_upsert
    _STATE.patched = False


@contextmanager
def stream_mode(
    *,
    target_batch_bytes: int = _DEFAULT_TARGET_BATCH_BYTES,
    max_batch_rows: int = 50_000,
    append_timeout_seconds: float | None = 120.0,
) -> Iterator[None]:
    """Enable Storage Write routing for in-memory inserts and upserts in this context."""

    options = StorageWritePatchOptions(
        target_batch_bytes=target_batch_bytes,
        max_batch_rows=max_batch_rows,
        append_timeout_seconds=append_timeout_seconds,
    )

    with _PATCH_LOCK:
        _STATE.active_contexts += 1
        _install_hook_locked()

    token = _ACTIVE_OPTIONS.set(options)
    try:
        yield
    finally:
        _ACTIVE_OPTIONS.reset(token)
        with _PATCH_LOCK:
            _STATE.active_contexts -= 1
            if _STATE.active_contexts < 0:
                _STATE.active_contexts = 0
                raise RuntimeError("stream_mode context tracking underflow")
            if _STATE.active_contexts == 0:
                _uninstall_hook_locked()


def is_patched() -> bool:
    """Return `True` while the BigQuery insert/upsert hooks are installed."""

    return _STATE.patched


__all__ = (
    "StorageWriteInsertError",
    "StorageWritePatchOptions",
    "is_patched",
    "stream_mode",
)
