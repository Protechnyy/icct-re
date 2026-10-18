import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from skill4re.extraction_schema import ExtractionSchemaSnapshot, ExtractionValidationError, default_config, validate_config


class ExtractionSchemaConflict(ExtractionValidationError):
    pass


class ExtractionSchemaStore:
    def __init__(self, storage_root):
        self.path = Path(storage_root) / "settings" / "extraction-schema.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=10)

    def get(self):
        if not self.path.exists():
            with self.lock:
                if not self.path.exists():
                    self._write(default_config())
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExtractionValidationError("extraction_schema", f"已保存配置无法解析：第 {exc.lineno} 行，第 {exc.colno} 列") from exc
        return validate_config(payload)

    def snapshot(self):
        snapshot = ExtractionSchemaSnapshot.from_payload(self.get())
        snapshot.require_active()
        return snapshot

    def save(self, payload):
        validate_config(payload)
        with self.lock:
            current = self.get()
            if payload["revision"] != current["revision"]:
                raise ExtractionSchemaConflict("revision", "配置已被更新，请重新读取后保存；当前版本为 " + str(current["revision"]))
            updated = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
            updated["revision"] = current["revision"] + 1
            self._write(updated)
        return updated

    def _write(self, payload):
        descriptor, filename = tempfile.mkstemp(prefix="extraction-schema-", suffix=".json", dir=self.path.parent)
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, allow_nan=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
