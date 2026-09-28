"""Persistent project, prompt, and recording metadata."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9-]{36}", value):
        raise ValueError("invalid project id")
    return value


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "trainer.sqlite3"
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, language TEXT NOT NULL,
                    espeak_voice TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL, mode TEXT NOT NULL, original_text TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_state (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(project_id, key)
                );
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready'
                );
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    prompt_id TEXT NOT NULL, raw_file TEXT NOT NULL, duration_seconds REAL NOT NULL,
                    audio_file TEXT, text_snapshot TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending', quality_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE RESTRICT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def has_project(self, project_id: str | None) -> bool:
        try:
            project_id = _safe_id(project_id)
        except (TypeError, ValueError):
            return False
        with self._connect() as db:
            return db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone() is not None

    def create_project(self, name: str, language: str = "pl_PL", espeak_voice: str = "pl") -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("project name must contain 1 to 80 characters")
        project_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
                (project_id, name, language, espeak_voice, _now()),
            )
        root = self.project_dir(project_id)
        for directory in ("source", "prompts", "recordings/raw", "recordings/normalized", "datasets", "runs", "models"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        return self.get_project(project_id)

    def project_dir(self, project_id: str) -> Path:
        return self.data_dir / "projects" / _safe_id(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM projects ORDER BY created_at")]
        return [self.get_project(project_id) for project_id in ids]

    def get_project(self, project_id: str) -> dict[str, Any]:
        project_id = _safe_id(project_id)
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            if row is None:
                raise KeyError("project not found")
            prompts = db.execute(
                "SELECT id, position, text, status FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position",
                (project_id,),
            ).fetchall()
            source = db.execute("SELECT filename, mode FROM sources WHERE project_id=?", (project_id,)).fetchone()
            counts = db.execute(
                "SELECT status, COUNT(*) AS n, SUM(duration_seconds) AS seconds FROM samples WHERE project_id=? GROUP BY status",
                (project_id,),
            ).fetchall()
        return {
            **dict(row),
            "prompts": [dict(item) for item in prompts],
            "source": dict(source) if source else None,
            "sample_counts": {item["status"]: {"count": item["n"], "seconds": item["seconds"] or 0} for item in counts},
        }

    def set_active_prompt(self, project_id: str, prompt_id: str | None) -> None:
        project_id = _safe_id(project_id)
        with self._connect() as db:
            if prompt_id:
                exists = db.execute("SELECT 1 FROM prompts WHERE id=? AND project_id=? AND status!='deleted'", (prompt_id, project_id)).fetchone()
                if not exists:
                    raise KeyError("active prompt not found")
                db.execute("INSERT INTO project_state VALUES (?, 'active_prompt_id', ?) ON CONFLICT(project_id,key) DO UPDATE SET value=excluded.value", (project_id, prompt_id))
            else:
                db.execute("DELETE FROM project_state WHERE project_id=? AND key='active_prompt_id'", (project_id,))

    def get_active_prompt(self, project_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM project_state WHERE project_id=? AND key='active_prompt_id'", (_safe_id(project_id),)).fetchone()
        return row[0] if row else None

    def import_prompts(self, project_id: str, filename: str, original_text: str, mode: str) -> list[str]:
        from app.text import parse_prompts

        project_id = _safe_id(project_id)
        if mode not in {"prose", "lines"}:
            raise ValueError("mode must be 'prose' or 'lines'")
        prompt_texts = parse_prompts(original_text, mode)
        root = self.project_dir(project_id)
        with self._connect() as db:
            if db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone() is None:
                raise KeyError("project not found")
            db.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?) ON CONFLICT(project_id) DO UPDATE SET filename=excluded.filename, mode=excluded.mode, original_text=excluded.original_text, imported_at=excluded.imported_at",
                (project_id, Path(filename).name[:120] or "pasted.txt", mode, original_text, _now()),
            )
            source_file = root / "source" / "original.txt"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text(original_text, encoding="utf-8")
            old = db.execute("SELECT id, position FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position", (project_id,)).fetchall()
            existing_ids = [item["id"] for item in old]
            # Re-import updates prompts in place where possible so sample foreign keys remain valid.
            for position, text in enumerate(prompt_texts):
                if position < len(old):
                    prompt_id = old[position]["id"]
                    db.execute("UPDATE prompts SET text=?, position=? WHERE id=?", (text, position, prompt_id))
                else:
                    prompt_id = str(uuid.uuid4())
                    db.execute("INSERT INTO prompts(id, project_id, position, text) VALUES (?, ?, ?, ?)", (prompt_id, project_id, position, text))
                existing_ids.append(prompt_id)
            for item in old[len(prompt_texts):]:
                db.execute("UPDATE prompts SET status='deleted' WHERE id=?", (item["id"],))
            active = db.execute("SELECT value FROM project_state WHERE project_id=? AND key='active_prompt_id'", (project_id,)).fetchone()
            valid_ids = {row[0] for row in db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted'", (project_id,))}
            if active and active[0] not in valid_ids:
                first = db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position LIMIT 1", (project_id,)).fetchone()
                if first:
                    db.execute("UPDATE project_state SET value=? WHERE project_id=? AND key='active_prompt_id'", (first[0], project_id))
                else:
                    db.execute("DELETE FROM project_state WHERE project_id=? AND key='active_prompt_id'", (project_id,))
            ids = [row[0] for row in db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position", (project_id,))]
            (root / "prompts" / "parsed.jsonl").write_text("".join(json.dumps({"id": prompt_id, "text": text}, ensure_ascii=False) + "\n" for prompt_id, text in zip(ids, prompt_texts)), encoding="utf-8")
            (root / "prompts" / "current.jsonl").write_text("".join(json.dumps({"id": prompt_id, "text": text}, ensure_ascii=False) + "\n" for prompt_id, text in zip(ids, prompt_texts)), encoding="utf-8")
        return ids

    def update_prompt(self, project_id: str, prompt_id: str, text: str) -> None:
        with self._connect() as db:
            result = db.execute("UPDATE prompts SET text=? WHERE id=? AND project_id=? AND status!='deleted'", (text.strip(), prompt_id, _safe_id(project_id)))
            if result.rowcount != 1:
                raise KeyError("prompt not found")
        self._write_current_prompts(project_id)

    def save_prompt_queue(self, project_id: str, rows: list[dict[str, str]]) -> None:
        project_id = _safe_id(project_id)
        with self._connect() as db:
            current = {row["id"] for row in db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted'", (project_id,))}
            db.execute("UPDATE prompts SET position=position+100000 WHERE project_id=? AND status!='deleted'", (project_id,))
            seen: set[str] = set()
            for position, row in enumerate(rows):
                prompt_id = str(row.get("id", "")).strip()
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                if not prompt_id or prompt_id not in current:
                    prompt_id = str(uuid.uuid4())
                    db.execute("INSERT INTO prompts(id, project_id, position, text) VALUES (?, ?, ?, ?)", (prompt_id, project_id, position, text))
                else:
                    db.execute("UPDATE prompts SET text=?, position=?, status='ready' WHERE id=?", (text, position, prompt_id))
                seen.add(prompt_id)
            for prompt_id in current - seen:
                db.execute("UPDATE prompts SET status='deleted' WHERE id=?", (prompt_id,))
            active = db.execute("SELECT value FROM project_state WHERE project_id=? AND key='active_prompt_id'", (project_id,)).fetchone()
            if active and active[0] not in seen:
                first = db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position LIMIT 1", (project_id,)).fetchone()
                if first:
                    db.execute("UPDATE project_state SET value=? WHERE project_id=? AND key='active_prompt_id'", (first[0], project_id))
                else:
                    db.execute("DELETE FROM project_state WHERE project_id=? AND key='active_prompt_id'", (project_id,))
        self._write_current_prompts(project_id)

    def move_prompt(self, project_id: str, index: int, direction: int) -> None:
        queue = self.get_project(project_id)["prompts"]
        target = index + direction
        if index < 0 or target < 0 or index >= len(queue) or target >= len(queue):
            return
        queue[index], queue[target] = queue[target], queue[index]
        self.save_prompt_queue(project_id, [{"id": item["id"], "text": item["text"]} for item in queue])

    def split_prompt(self, project_id: str, index: int, left: str, right: str) -> None:
        queue = self.get_project(project_id)["prompts"]
        if index < 0 or index >= len(queue) or not left.strip() or not right.strip():
            raise ValueError("select a prompt and provide non-empty text on both sides")
        queue[index]["text"] = left.strip()
        queue.insert(index + 1, {"id": "", "text": right.strip()})
        self.save_prompt_queue(project_id, [{"id": item.get("id", ""), "text": item["text"]} for item in queue])

    def merge_prompts(self, project_id: str, index: int) -> None:
        queue = self.get_project(project_id)["prompts"]
        if index < 0 or index + 1 >= len(queue):
            raise ValueError("select a prompt with an adjacent prompt after it")
        queue[index]["text"] = queue[index]["text"].rstrip() + " " + queue[index + 1]["text"].lstrip()
        del queue[index + 1]
        self.save_prompt_queue(project_id, [{"id": item["id"], "text": item["text"]} for item in queue])

    def delete_prompt(self, project_id: str, prompt_id: str) -> None:
        with self._connect() as db:
            result = db.execute("UPDATE prompts SET status='deleted' WHERE id=? AND project_id=? AND status!='deleted'", (prompt_id, _safe_id(project_id)))
            if result.rowcount != 1:
                raise KeyError("prompt not found")
            db.execute("UPDATE prompts SET position=position+100000 WHERE project_id=? AND position>(SELECT position FROM prompts WHERE id=?)", (_safe_id(project_id), prompt_id))
            db.execute("UPDATE prompts SET position=position-100001 WHERE project_id=? AND position>=100000", (_safe_id(project_id),))
            active = db.execute("SELECT value FROM project_state WHERE project_id=? AND key='active_prompt_id'", (_safe_id(project_id),)).fetchone()
            if active and active[0] == prompt_id:
                first = db.execute("SELECT id FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position LIMIT 1", (_safe_id(project_id),)).fetchone()
                if first:
                    db.execute("UPDATE project_state SET value=? WHERE project_id=? AND key='active_prompt_id'", (first[0], _safe_id(project_id)))
                else:
                    db.execute("DELETE FROM project_state WHERE project_id=? AND key='active_prompt_id'", (_safe_id(project_id),))
        self._write_current_prompts(project_id)

    def add_sample(self, project_id: str, prompt_id: str, raw_file: str, duration_seconds: float, status: str = "accepted", quality: dict[str, Any] | None = None, audio_file: str | None = None) -> str:
        if duration_seconds < 0:
            raise ValueError("duration must be non-negative")
        sample_id = str(uuid.uuid4())
        with self._connect() as db:
            prompt = db.execute("SELECT text FROM prompts WHERE id=? AND project_id=?", (prompt_id, _safe_id(project_id))).fetchone()
            if prompt is None:
                raise KeyError("prompt not found")
            db.execute("UPDATE samples SET status='superseded' WHERE prompt_id=? AND status IN ('accepted','review','pending')", (prompt_id,))
            db.execute(
                "INSERT INTO samples(id, project_id, prompt_id, raw_file, duration_seconds, audio_file, text_snapshot, status, quality_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, _safe_id(project_id), prompt_id, raw_file, duration_seconds, audio_file, prompt["text"], status, json.dumps(quality or {}), _now()),
            )
        return sample_id

    def list_samples(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT s.*, s.text_snapshot AS text FROM samples s WHERE s.project_id=? ORDER BY s.created_at",
                (_safe_id(project_id),),
            ).fetchall()
        return [{**dict(row), "quality": json.loads(row["quality_json"])} for row in rows]

    def set_sample_status(self, project_id: str, sample_id: str, status: str) -> None:
        if status not in {"accepted", "review", "rejected"}:
            raise ValueError("invalid sample status")
        with self._connect() as db:
            result = db.execute("UPDATE samples SET status=? WHERE id=? AND project_id=?", (status, sample_id, _safe_id(project_id)))
            if result.rowcount != 1:
                raise KeyError("sample not found")

    def _write_current_prompts(self, project_id: str) -> None:
        root = self.project_dir(project_id)
        with self._connect() as db:
            rows = db.execute("SELECT id, text FROM prompts WHERE project_id=? AND status!='deleted' ORDER BY position", (_safe_id(project_id),)).fetchall()
        content = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
        (root / "prompts" / "current.jsonl").write_text(content, encoding="utf-8")
