from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import ibis
import ibis.expr.types as ir
import pyarrow as pa
import pytest
from google.auth.credentials import AnonymousCredentials

import ibis_stream.bigquery as mod

pytestmark = [pytest.mark.unit, pytest.mark.bigquery]


class _FakeBackend:
    def __init__(self) -> None:
        self.current_catalog = "project_default"
        self.current_database = "dataset_default"
        self.client = SimpleNamespace(_credentials=AnonymousCredentials())
        self.truncate_calls: list[tuple[str, Any]] = []
        self.create_table_calls: list[tuple[str, ibis.Schema, str | None]] = []
        self.drop_table_calls: list[tuple[str, Any, bool]] = []
        self.table_schemas: dict[tuple[str | None, str], ibis.Schema] = {}

    @staticmethod
    def _normalize_db_key(database: Any) -> str | None:
        if database is None:
            return None
        if isinstance(database, tuple):
            if len(database) == 2:
                return f"{database[0]}.{database[1]}"
            if len(database) == 1:
                return str(database[0])
            return ".".join(str(part) for part in database)
        return str(database)

    def _to_sqlglot_table(self, database: str | None) -> str | None:
        return database

    def _to_catalog_db_tuple(self, table_loc: str | None) -> tuple[str | None, str | None]:
        if table_loc is None:
            return None, None
        parts = table_loc.split(".")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, parts[0]

    def truncate_table(self, name: str, *, database: str | None = None) -> None:
        self.truncate_calls.append((name, database))

    def create_table(
        self,
        name: str,
        /,
        obj: Any = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        **_: Any,
    ) -> Any:
        del obj, temp, overwrite
        if schema is None:
            raise AssertionError("schema must be provided in unit tests")
        self.create_table_calls.append((name, schema, database))
        db_key = self._normalize_db_key(database)
        self.table_schemas[(db_key, name)] = schema
        return ibis.table(schema, name=name)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        self.drop_table_calls.append((name, database, force))
        db_key = self._normalize_db_key(database)
        self.table_schemas.pop((db_key, name), None)

    def table(self, name: str, /, *, database: str | None = None) -> ir.Table:
        db_key = self._normalize_db_key(database)
        schema = self.table_schemas.get((db_key, name))
        if schema is None and isinstance(database, tuple) and len(database) == 2:
            schema = self.table_schemas.get((str(database[1]), name))
        if schema is None:
            schema = self.table_schemas.get((None, name))
        if schema is None:
            raise KeyError((database, name))
        return ibis.table(schema, name=name)


@contextmanager
def _active_stream_options(
    *,
    target_batch_bytes: int = 128,
    max_batch_rows: int = 50_000,
    append_timeout_seconds: float | None = 120.0,
) -> Any:
    options = mod.StorageWritePatchOptions(
        target_batch_bytes=target_batch_bytes,
        max_batch_rows=max_batch_rows,
        append_timeout_seconds=append_timeout_seconds,
    )
    token = mod._ACTIVE_OPTIONS.set(options)
    try:
        yield options
    finally:
        mod._ACTIVE_OPTIONS.reset(token)


def test_coerce_to_arrow_input_supports_pyarrow_types() -> None:
    arrow = pa.table({"x": [10, 20]})
    assert mod._coerce_to_arrow_input(arrow) is arrow

    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("a", pa.int64())]),
        [pa.record_batch([pa.array([1, 2])], names=["a"])],
    )
    assert mod._coerce_to_arrow_input(reader) is reader

    assert mod._coerce_to_arrow_input([{"x": 1}]) is None


def test_resolve_table_uses_database_defaults_when_name_unqualified() -> None:
    backend = _FakeBackend()
    resolved = mod._resolve_table(backend, name="target", database="proj.ds")
    assert resolved.project == "proj"
    assert resolved.dataset == "ds"
    assert resolved.table == "target"


def test_resolve_table_prefers_name_qualification() -> None:
    backend = _FakeBackend()
    resolved = mod._resolve_table(backend, name="proj2.ds2.table2", database="proj.ds")
    assert resolved.project == "proj2"
    assert resolved.dataset == "ds2"
    assert resolved.table == "table2"


def test_resolve_table_two_part_name_uses_default_project() -> None:
    backend = _FakeBackend()
    resolved = mod._resolve_table(backend, name="ds2.table2", database="proj.ds")
    assert resolved.project == "proj"
    assert resolved.dataset == "ds2"
    assert resolved.table == "table2"


def test_resolve_table_handles_quoted_fully_qualified_name() -> None:
    backend = _FakeBackend()
    resolved = mod._resolve_table(backend, name="`proj2.ds2.table2`", database="proj.ds")
    assert resolved.project == "proj2"
    assert resolved.dataset == "ds2"
    assert resolved.table == "table2"


def test_resolve_table_rejects_overspecified_name() -> None:
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="Unable to resolve BigQuery table path"):
        mod._resolve_table(backend, name="proj.ds.tbl.extra", database="proj.ds")


def test_resolve_table_normalizes_dataset_prefixed_with_project() -> None:
    backend = _FakeBackend()
    backend.current_catalog = "proj"
    backend.current_database = "proj.ds"

    resolved = mod._resolve_table(backend, name="target", database=None)
    assert resolved.project == "proj"
    assert resolved.dataset == "ds"
    assert resolved.table == "target"


def test_patched_insert_falls_back_to_original_for_non_in_memory_ibis_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    calls: list[tuple[str, Any, str | None, bool]] = []

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        calls.append((name, obj, database, overwrite))

    monkeypatch.setattr(mod._STATE, "original_insert", _original)

    expr = ibis.table([("a", "int64")], name="remote_table")
    mod._patched_insert(backend, "target", expr, database="proj.ds", overwrite=False)
    assert len(calls) == 1
    assert calls[0][0] == "target"
    assert calls[0][1] is expr


def test_patched_insert_routes_pandas_to_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    backend = _FakeBackend()
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        raise AssertionError("original insert should not be called for pandas inserts")

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["target"] = target
        seen["rows"] = arrow_input.num_rows
        seen["options"] = options

    monkeypatch.setattr(mod._STATE, "original_insert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)

    dataframe = pandas.DataFrame({"a": [1, 2, 3]})
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", dataframe, database="proj.ds", overwrite=True)

    assert seen["target"] == mod._ResolvedBigQueryTable("proj", "ds", "target")
    assert seen["rows"] == 3
    assert seen["options"].target_batch_bytes == 128
    assert backend.truncate_calls == [("target", ("proj", "ds"))]


def test_patched_insert_overwrite_uses_resolved_default_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    backend = _FakeBackend()

    monkeypatch.setattr(mod._STATE, "original_insert", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", lambda *_args, **_kwargs: None)

    dataframe = pandas.DataFrame({"a": [1]})
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", dataframe, overwrite=True)

    assert backend.truncate_calls == [("target", ("project_default", "dataset_default"))]


def test_patched_insert_routes_list_dict_via_memtable_to_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        raise AssertionError("original insert should not be called for list/dict inserts")

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["target"] = target
        seen["rows"] = arrow_input.num_rows
        seen["cols"] = arrow_input.column_names
        seen["options"] = options

    monkeypatch.setattr(mod._STATE, "original_insert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)

    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", rows, database="proj.ds", overwrite=False)

    assert seen["target"] == mod._ResolvedBigQueryTable("proj", "ds", "target")
    assert seen["rows"] == 2
    assert seen["cols"] == ["a", "b"]
    assert seen["options"].target_batch_bytes == 128


def test_patched_insert_routes_ibis_memtable_to_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        raise AssertionError("original insert should not be called for memtable inserts")

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["target"] = target
        seen["rows"] = arrow_input.num_rows
        seen["options"] = options

    monkeypatch.setattr(mod._STATE, "original_insert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)

    expr = ibis.memtable({"a": [1, 2, 3]})
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", expr, database="proj.ds", overwrite=False)

    assert seen["target"] == mod._ResolvedBigQueryTable("proj", "ds", "target")
    assert seen["rows"] == 3
    assert seen["options"].target_batch_bytes == 128


def test_patched_insert_routes_polars_via_memtable_to_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polars = pytest.importorskip("polars")
    backend = _FakeBackend()
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        raise AssertionError("original insert should not be called for polars inserts")

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["target"] = target
        seen["rows"] = arrow_input.num_rows
        seen["options"] = options

    monkeypatch.setattr(mod._STATE, "original_insert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)

    dataframe = polars.DataFrame({"a": [1, 2, 3]})
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", dataframe, database="proj.ds", overwrite=False)

    assert seen["target"] == mod._ResolvedBigQueryTable("proj", "ds", "target")
    assert seen["rows"] == 3
    assert seen["options"].target_batch_bytes == 128


def test_patched_insert_routes_record_batch_reader_to_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        raise AssertionError("original insert should not be called for Arrow reader inserts")

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["target"] = target
        seen["is_reader"] = isinstance(arrow_input, pa.RecordBatchReader)
        seen["options"] = options

    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("a", pa.int64())]),
        [pa.record_batch([pa.array([1, 2, 3])], names=["a"])],
    )

    monkeypatch.setattr(mod._STATE, "original_insert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)

    with _active_stream_options(target_batch_bytes=128):
        mod._patched_insert(backend, "target", reader, database="proj.ds", overwrite=False)

    assert seen["target"] == mod._ResolvedBigQueryTable("proj", "ds", "target")
    assert seen["is_reader"] is True
    assert seen["options"].target_batch_bytes == 128


def test_patched_upsert_falls_back_to_original_when_stream_mode_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    calls: list[tuple[str, Any, str, str | None]] = []

    def _original(
        self: Any,
        name: str,
        obj: Any,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        calls.append((name, obj, on, database))

    monkeypatch.setattr(mod._STATE, "original_upsert", _original)
    payload = [{"id": 1, "payload": "a"}]
    mod._patched_upsert(backend, "target", payload, "id", database="proj.ds")
    assert calls == [("target", payload, "id", "proj.ds")]


def test_patched_upsert_falls_back_to_original_for_non_in_memory_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    calls: list[tuple[str, Any, str, str | None]] = []

    def _original(
        self: Any,
        name: str,
        obj: Any,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        calls.append((name, obj, on, database))

    monkeypatch.setattr(mod._STATE, "original_upsert", _original)
    expr = ibis.table([("id", "int64"), ("payload", "string")], name="remote_table")

    with _active_stream_options(target_batch_bytes=128):
        mod._patched_upsert(backend, "target", expr, "id", database="proj.ds")

    assert len(calls) == 1
    assert calls[0][0] == "target"
    assert calls[0][1] is expr
    assert calls[0][2] == "id"
    assert calls[0][3] == "proj.ds"


def test_patched_upsert_routes_pandas_via_staging_storage_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")
    backend = _FakeBackend()
    backend.table_schemas[("proj.ds", "target")] = ibis.schema(
        {
            "id": "int64",
            "payload": "string",
        }
    )
    seen: dict[str, Any] = {}

    def _original(
        self: Any,
        name: str,
        obj: Any,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        seen["upsert_name"] = name
        seen["upsert_obj"] = obj
        seen["upsert_on"] = on
        seen["upsert_database"] = database

    def _fake_append(
        self: Any,
        *,
        target: mod._ResolvedBigQueryTable,
        arrow_input: mod._ArrowInput,
        options: mod.StorageWritePatchOptions,
    ) -> None:
        seen["append_target"] = target
        seen["append_rows"] = arrow_input.num_rows
        seen["append_options"] = options

    monkeypatch.setattr(mod._STATE, "original_upsert", _original)
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", _fake_append)
    monkeypatch.setattr(mod, "_new_staging_table_name", lambda: "_ibis_stream_upsert_fixed")

    dataframe = pandas.DataFrame(
        {
            "id": [1, 2, 3],
            "payload": ["a", "b", "c"],
        }
    )
    with _active_stream_options(target_batch_bytes=128):
        mod._patched_upsert(backend, "target", dataframe, "id", database="proj.ds")

    assert backend.create_table_calls
    create_name, create_schema, create_db = backend.create_table_calls[0]
    assert create_name == "_ibis_stream_upsert_fixed"
    assert create_db == "ds"
    assert tuple(create_schema.names) == ("id", "payload")

    assert seen["append_target"] == mod._ResolvedBigQueryTable(
        "proj",
        "ds",
        "_ibis_stream_upsert_fixed",
    )
    assert seen["append_rows"] == 3
    assert seen["append_options"].target_batch_bytes == 128

    assert seen["upsert_name"] == "target"
    assert isinstance(seen["upsert_obj"], ir.Table)
    assert seen["upsert_on"] == "id"
    assert seen["upsert_database"] == "proj.ds"

    assert backend.drop_table_calls == [
        (
            "_ibis_stream_upsert_fixed",
            ("proj", "ds"),
            True,
        )
    ]


def test_patched_upsert_rejects_missing_key_in_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    backend.table_schemas[("proj.ds", "target")] = ibis.schema({"payload": "string"})

    monkeypatch.setattr(mod._STATE, "original_upsert", lambda *_args, **_kwargs: None)

    rows = [{"id": 1, "payload": "a"}]
    with (
        _active_stream_options(target_batch_bytes=128),
        pytest.raises(ValueError, match="not found in target table"),
    ):
        mod._patched_upsert(backend, "target", rows, "id", database="proj.ds")

    assert backend.create_table_calls == []


def test_patched_upsert_rejects_missing_key_in_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    backend.table_schemas[("proj.ds", "target")] = ibis.schema(
        {
            "id": "int64",
            "payload": "string",
        }
    )

    monkeypatch.setattr(mod._STATE, "original_upsert", lambda *_args, **_kwargs: None)

    rows = [{"payload": "a"}]
    with (
        _active_stream_options(target_batch_bytes=128),
        pytest.raises(ValueError, match="not found in source data"),
    ):
        mod._patched_upsert(backend, "target", rows, "id", database="proj.ds")

    assert backend.create_table_calls == []


def test_upsert_via_storage_write_preserves_error_when_drop_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    backend.table_schemas[("proj.ds", "target")] = ibis.schema(
        {
            "id": "int64",
            "payload": "string",
        }
    )
    arrow_table = pa.table({"id": [1], "payload": ["bad"]})

    monkeypatch.setattr(mod, "_new_staging_table_name", lambda: "_ibis_stream_upsert_fixed")

    def _original(
        self: Any,
        name: str,
        obj: Any,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        del self, name, obj, on, database

    def _drop_raising(
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        del name, database, force
        raise RuntimeError("drop failed")

    monkeypatch.setattr(mod._STATE, "original_upsert", _original)
    monkeypatch.setattr(backend, "drop_table", _drop_raising)
    monkeypatch.setattr(
        mod,
        "_append_arrow_via_storage_write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            mod.StorageWriteInsertError("write failed")
        ),
    )

    with pytest.raises(mod.StorageWriteInsertError, match="write failed") as caught:
        mod._upsert_via_storage_write(
            backend,
            name="target",
            on="id",
            database="proj.ds",
            arrow_input=arrow_table,
            options=mod.StorageWritePatchOptions(target_batch_bytes=128),
        )

    notes = getattr(caught.value, "__notes__", [])
    assert any("drop failed" in note for note in notes)


def test_upsert_via_storage_write_raises_when_drop_fails_without_prior_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    backend.table_schemas[("proj.ds", "target")] = ibis.schema(
        {
            "id": "int64",
            "payload": "string",
        }
    )
    arrow_table = pa.table({"id": [1], "payload": ["ok"]})

    monkeypatch.setattr(mod, "_new_staging_table_name", lambda: "_ibis_stream_upsert_fixed")
    monkeypatch.setattr(mod._STATE, "original_upsert", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        backend,
        "drop_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("drop failed")),
    )
    monkeypatch.setattr(mod, "_append_arrow_via_storage_write", lambda *_args, **_kwargs: None)

    with pytest.raises(mod.StorageWriteInsertError, match="Failed to drop BigQuery upsert staging"):
        mod._upsert_via_storage_write(
            backend,
            name="target",
            on="id",
            database="proj.ds",
            arrow_input=arrow_table,
            options=mod.StorageWritePatchOptions(target_batch_bytes=128),
        )


def test_append_arrow_via_storage_write_emits_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    arrow_table = pa.table({"a": list(range(7))})
    requests: list[Any] = []

    class _FakeWriteClient:
        @staticmethod
        def table_path(project: str, dataset: str, table: str) -> str:
            return f"projects/{project}/datasets/{dataset}/tables/{table}"

    class _FakeFuture:
        @staticmethod
        def result(timeout: float | None = None) -> Any:
            del timeout
            return SimpleNamespace(row_errors=[])

    class _FakeAppendRowsStream:
        def __init__(self, client: Any, initial_request: Any) -> None:
            self.client = client
            self.initial_request = initial_request
            self.closed = False
            requests.append(("init", initial_request))

        def send(self, request: Any) -> _FakeFuture:
            requests.append(("append", request))
            return _FakeFuture()

        def close(self) -> None:
            self.closed = True

    batch_1 = pa.record_batch([pa.array([0, 1])], names=["a"])
    batch_2 = pa.record_batch([pa.array([2, 3, 4, 5, 6])], names=["a"])

    def _fake_iter_storage_write_batches(
        table: pa.Table,
        *,
        options: mod.StorageWritePatchOptions,
    ) -> Any:
        assert table.num_rows == 7
        assert options.append_timeout_seconds == 5.0
        yield batch_1, b"batch_1"
        yield batch_2, b"batch_2"

    monkeypatch.setattr(mod, "_get_or_create_write_client", lambda _backend: _FakeWriteClient())
    monkeypatch.setattr(mod, "AppendRowsStream", _FakeAppendRowsStream)
    monkeypatch.setattr(mod, "_iter_storage_write_batches", _fake_iter_storage_write_batches)

    mod._append_arrow_via_storage_write(
        backend,
        target=mod._ResolvedBigQueryTable("p", "d", "t"),
        arrow_input=arrow_table,
        options=mod.StorageWritePatchOptions(append_timeout_seconds=5.0),
    )

    init_kind, init_request = requests[0]
    assert init_kind == "init"
    assert init_request.write_stream == "projects/p/datasets/d/tables/t/streams/_default"

    append_requests = [req for kind, req in requests if kind == "append"]
    assert len(append_requests) == 2
    assert [req.arrow_rows.rows.row_count for req in append_requests] == [2, 5]
    assert [req.arrow_rows.rows.serialized_record_batch for req in append_requests] == [
        b"batch_1",
        b"batch_2",
    ]


def test_iter_storage_write_batches_splits_by_serialized_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_table = pa.table({"a": list(range(7))})

    def _fake_serialize(batch: pa.RecordBatch) -> bytes:
        return b"x" * (batch.num_rows * 10)

    monkeypatch.setattr(mod, "_serialize_record_batch", _fake_serialize)

    options = mod.StorageWritePatchOptions(
        target_batch_bytes=30,
        max_batch_rows=100,
    )
    batches = list(mod._iter_storage_write_batches(arrow_table, options=options))

    assert [batch.num_rows for batch, _ in batches] == [3, 2, 2]
    assert all(len(serialized) <= 30 for _, serialized in batches)


def test_append_arrow_via_storage_write_raises_on_row_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    arrow_table = pa.table({"a": [1]})

    class _FakeWriteClient:
        @staticmethod
        def table_path(project: str, dataset: str, table: str) -> str:
            return f"projects/{project}/datasets/{dataset}/tables/{table}"

    class _FakeFuture:
        @staticmethod
        def result(timeout: float | None = None) -> Any:
            del timeout
            row_error = SimpleNamespace(index=0, message="bad row")
            return SimpleNamespace(row_errors=[row_error])

    class _FakeAppendRowsStream:
        def __init__(self, client: Any, initial_request: Any) -> None:
            del client, initial_request

        def send(self, request: Any) -> _FakeFuture:
            del request
            return _FakeFuture()

        def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "_get_or_create_write_client", lambda _backend: _FakeWriteClient())
    monkeypatch.setattr(mod, "AppendRowsStream", _FakeAppendRowsStream)

    with pytest.raises(mod.StorageWriteInsertError, match="bad row"):
        mod._append_arrow_via_storage_write(
            backend,
            target=mod._ResolvedBigQueryTable("p", "d", "t"),
            arrow_input=arrow_table,
            options=mod.StorageWritePatchOptions(),
        )


def test_append_arrow_via_storage_write_raises_when_close_fails_without_prior_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    arrow_table = pa.table({"a": [1]})

    class _FakeWriteClient:
        @staticmethod
        def table_path(project: str, dataset: str, table: str) -> str:
            return f"projects/{project}/datasets/{dataset}/tables/{table}"

    class _FakeFuture:
        @staticmethod
        def result(timeout: float | None = None) -> Any:
            del timeout
            return SimpleNamespace(row_errors=[])

    class _FakeAppendRowsStream:
        def __init__(self, client: Any, initial_request: Any) -> None:
            del client, initial_request

        def send(self, request: Any) -> _FakeFuture:
            del request
            return _FakeFuture()

        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(mod, "_get_or_create_write_client", lambda _backend: _FakeWriteClient())
    monkeypatch.setattr(mod, "AppendRowsStream", _FakeAppendRowsStream)

    with pytest.raises(mod.StorageWriteInsertError, match="Failed to close"):
        mod._append_arrow_via_storage_write(
            backend,
            target=mod._ResolvedBigQueryTable("p", "d", "t"),
            arrow_input=arrow_table,
            options=mod.StorageWritePatchOptions(),
        )


def test_append_arrow_via_storage_write_preserves_write_error_when_close_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    arrow_table = pa.table({"a": [1]})

    class _FakeWriteClient:
        @staticmethod
        def table_path(project: str, dataset: str, table: str) -> str:
            return f"projects/{project}/datasets/{dataset}/tables/{table}"

    class _FakeFuture:
        @staticmethod
        def result(timeout: float | None = None) -> Any:
            del timeout
            row_error = SimpleNamespace(index=0, message="bad row")
            return SimpleNamespace(row_errors=[row_error])

    class _FakeAppendRowsStream:
        def __init__(self, client: Any, initial_request: Any) -> None:
            del client, initial_request

        def send(self, request: Any) -> _FakeFuture:
            del request
            return _FakeFuture()

        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(mod, "_get_or_create_write_client", lambda _backend: _FakeWriteClient())
    monkeypatch.setattr(mod, "AppendRowsStream", _FakeAppendRowsStream)

    with pytest.raises(mod.StorageWriteInsertError, match="bad row") as caught:
        mod._append_arrow_via_storage_write(
            backend,
            target=mod._ResolvedBigQueryTable("p", "d", "t"),
            arrow_input=arrow_table,
            options=mod.StorageWritePatchOptions(),
        )

    notes = getattr(caught.value, "__notes__", [])
    assert any("close failed" in note for note in notes)


def test_storage_write_patch_options_rejects_oversized_target_batch_bytes() -> None:
    with pytest.raises(ValueError, match="must be <="):
        mod.StorageWritePatchOptions(target_batch_bytes=10 * 1024 * 1024)


def test_stream_mode_lifecycle_and_nested_option_restore() -> None:
    from ibis.backends.bigquery import Backend

    backend_insert_before = Backend.insert
    backend_upsert_before = Backend.upsert
    state_insert_before = mod._STATE.original_insert
    state_upsert_before = mod._STATE.original_upsert
    state_patched_before = mod._STATE.patched
    state_active_contexts_before = mod._STATE.active_contexts
    active_options_before = mod._ACTIVE_OPTIONS.get()
    try:
        mod._STATE.original_insert = backend_insert_before
        mod._STATE.original_upsert = backend_upsert_before
        mod._STATE.patched = False
        mod._STATE.active_contexts = 0
        assert not mod.is_patched()

        with mod.stream_mode(
            target_batch_bytes=256, max_batch_rows=100, append_timeout_seconds=1.5
        ):
            assert mod.is_patched()
            assert Backend.insert is mod._patched_insert
            assert Backend.upsert is mod._patched_upsert
            assert mod._STATE.active_contexts == 1
            outer_options = mod._ACTIVE_OPTIONS.get()
            assert outer_options is not None
            assert outer_options.target_batch_bytes == 256
            assert outer_options.max_batch_rows == 100
            assert outer_options.append_timeout_seconds == 1.5

            with mod.stream_mode(
                target_batch_bytes=512,
                max_batch_rows=200,
                append_timeout_seconds=2.5,
            ):
                assert mod.is_patched()
                assert Backend.insert is mod._patched_insert
                assert Backend.upsert is mod._patched_upsert
                assert mod._STATE.active_contexts == 2
                inner_options = mod._ACTIVE_OPTIONS.get()
                assert inner_options is not None
                assert inner_options.target_batch_bytes == 512
                assert inner_options.max_batch_rows == 200
                assert inner_options.append_timeout_seconds == 2.5

            restored_outer_options = mod._ACTIVE_OPTIONS.get()
            assert restored_outer_options is outer_options
            assert mod._STATE.active_contexts == 1

        assert not mod.is_patched()
        assert mod._STATE.active_contexts == 0
        assert Backend.insert is backend_insert_before
        assert Backend.upsert is backend_upsert_before
    finally:
        mod._STATE.original_insert = state_insert_before
        mod._STATE.original_upsert = state_upsert_before
        mod._STATE.patched = state_patched_before
        mod._STATE.active_contexts = state_active_contexts_before
        mod._ACTIVE_OPTIONS.set(active_options_before)
        if mod._STATE.patched:
            Backend.insert = mod._patched_insert
            Backend.upsert = mod._patched_upsert
        elif state_insert_before is not None and state_upsert_before is not None:
            Backend.insert = state_insert_before
            Backend.upsert = state_upsert_before
        else:
            Backend.insert = backend_insert_before
            Backend.upsert = backend_upsert_before
