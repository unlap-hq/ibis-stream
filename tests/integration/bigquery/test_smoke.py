from __future__ import annotations

import pytest
from ibis.backends.bigquery import Backend

import ibis_stream.bigquery as mod

pytestmark = [pytest.mark.integration, pytest.mark.bigquery]


def test_patch_install_uninstall_smoke() -> None:
    backend_insert_before = Backend.insert
    backend_upsert_before = Backend.upsert
    state_insert_before = mod._STATE.original_insert
    state_upsert_before = mod._STATE.original_upsert
    state_patched_before = mod._STATE.patched
    state_active_contexts_before = mod._STATE.active_contexts
    active_options_before = mod._ACTIVE_OPTIONS.get()
    try:
        with mod.stream_mode(target_batch_bytes=256 * 1024, max_batch_rows=128):
            assert mod.is_patched()
            assert Backend.insert is mod._patched_insert
            assert Backend.upsert is mod._patched_upsert
            assert mod._STATE.active_contexts == 1
            options = mod._ACTIVE_OPTIONS.get()
            assert options is not None
            assert options.target_batch_bytes == 256 * 1024
            assert options.max_batch_rows == 128
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
