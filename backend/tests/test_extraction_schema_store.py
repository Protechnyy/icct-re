import json
from concurrent.futures import ProcessPoolExecutor

import pytest

from app.extraction_schema_store import ExtractionSchemaConflict, ExtractionSchemaStore
from skill4re.extraction_schema import ExtractionValidationError


def save_revision(arguments):
    root, payload = arguments
    store = ExtractionSchemaStore(root)
    try:
        return store.save(payload)["revision"]
    except ExtractionSchemaConflict:
        return "conflict"


def test_atomic_save_and_process_conflict(tmp_path):
    store = ExtractionSchemaStore(tmp_path)
    initial = store.get()
    with ProcessPoolExecutor(max_workers=2) as executor:
        revisions = list(executor.map(save_revision, [(tmp_path, initial), (tmp_path, initial)]))
    assert sorted(map(str, revisions)) == ["2", "conflict"]
    assert store.get()["revision"] == 2
    assert not list(store.path.parent.glob("extraction-schema-*.json"))


def test_invalid_save_preserves_file_and_snapshot(tmp_path):
    store = ExtractionSchemaStore(tmp_path)
    snapshot = store.snapshot()
    before = store.path.read_bytes()
    payload = store.get()
    payload["entity_types"][0]["parent_id"] = payload["entity_types"][0]["id"]
    with pytest.raises(ExtractionValidationError):
        store.save(payload)
    assert store.path.read_bytes() == before
    updated = store.get()
    updated["entity_types"][0]["name"] = "新增名称"
    store.save(updated)
    assert snapshot.payload()["entity_types"][0]["name"] == "指挥机构"
    assert store.snapshot().revision == 2


def test_damaged_saved_file_is_preserved(tmp_path):
    store = ExtractionSchemaStore(tmp_path)
    store.get()
    store.path.write_text("{", encoding="utf-8")
    with pytest.raises(ExtractionValidationError, match="无法解析"):
        store.get()
    assert store.path.read_text(encoding="utf-8") == "{"


def test_filesystem_write_failure_preserves_saved_config(tmp_path):
    store = ExtractionSchemaStore(tmp_path)
    payload = store.get()
    original = store.path.read_bytes()
    mode = store.path.parent.stat().st_mode
    store.path.parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            store.save(payload)
    finally:
        store.path.parent.chmod(mode)
    assert store.path.read_bytes() == original
    assert not list(store.path.parent.glob("extraction-schema-*.json"))
