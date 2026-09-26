"""Durable storage for served agent runs (SQLite, stdlib only).

The graph's own checkpoints live in the same file, written by LangGraph's
SqliteSaver. This store keeps what LangGraph doesn't: the run's lifecycle
status, the human's decision, and the state the tool runtime holds outside
the graph (budget counters, idempotency ledger and breaker counts, the
simulated backend) plus the audit log. Each is written as it changes, so a
crash loses at most the call in flight.

Status lifecycle:

    starting ──> awaiting_approval ──> executing ──> completed | escalated
        │              │                   │
        │              └─> expired         └─> interrupted ──(recover)──> executing
        └──────────────────────────────────────> completed | escalated | interrupted

A run is claimed for execution with a compare-and-set on its status, so
across threads *and* processes exactly one approval request executes it.
Timestamps are wall-clock seconds (time.time()), comparable across processes.
"""
import json
import sqlite3
import threading
from pathlib import Path

FINAL = ("completed", "escalated", "expired")
RUNNING = ("starting", "executing")

_JSON_COLUMNS = ("proposal", "decision", "response", "budget", "executor", "infra")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id  TEXT PRIMARY KEY,
    alert      TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL,            -- awaiting_approval only: when the proposal goes stale
    proposal   TEXT,            -- the proposal at the approval gate
    root_cause TEXT,
    decision   TEXT,            -- {"approved", "args_hash", "approver"}, set when claimed
    response   TEXT,            -- the summary returned once the run settles
    budget     TEXT,            -- RunBudget.to_dict()
    executor   TEXT,            -- ToolExecutor.state(): idempotency ledger + breaker counts
    infra      TEXT             -- InfraSimulator.to_dict()
);
CREATE TABLE IF NOT EXISTS audit (
    thread_id TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    record    TEXT NOT NULL,
    PRIMARY KEY (thread_id, seq)
);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, updated_at);
"""


class RunStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Autocommit; each statement is its own transaction. WAL lets readers
        # (GET /runs) proceed while a run is writing.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()  # one statement at a time on this connection

    def _exec(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def create(self, thread_id: str, alert: str, now: float) -> None:
        self._exec("INSERT INTO runs (thread_id, alert, status, created_at, updated_at) VALUES (?, ?, 'starting', ?, ?)",
                   (thread_id, alert, now, now))

    def get(self, thread_id: str) -> dict | None:
        row = self._exec("SELECT * FROM runs WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        for col in _JSON_COLUMNS:
            if run[col] is not None:
                run[col] = json.loads(run[col])
        return run

    def update(self, thread_id: str, now: float, **fields) -> None:
        """Set fields and bump updated_at, which doubles as the run's
        heartbeat: a running run writes on every budget charge."""
        cols = [f"{k} = ?" for k in fields] + ["updated_at = ?"]
        vals = [json.dumps(v) if k in _JSON_COLUMNS and v is not None else v for k, v in fields.items()]
        self._exec(f"UPDATE runs SET {', '.join(cols)} WHERE thread_id = ?", (*vals, now, thread_id))

    def transition(self, thread_id: str, from_status: str, to_status: str, now: float, **fields) -> bool:
        """Compare-and-set the status. True if this caller won: the run was
        in `from_status` and is now in `to_status`."""
        cols = ["status = ?"] + [f"{k} = ?" for k in fields] + ["updated_at = ?"]
        vals = [to_status] + [json.dumps(v) if k in _JSON_COLUMNS and v is not None else v
                              for k, v in fields.items()] + [now]
        cur = self._exec(f"UPDATE runs SET {', '.join(cols)} WHERE thread_id = ? AND status = ?",
                         (*vals, thread_id, from_status))
        return cur.rowcount == 1

    def append_audit(self, thread_id: str, record: dict) -> None:
        self._exec("INSERT INTO audit (thread_id, seq, record) VALUES "
                   "(?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM audit WHERE thread_id = ?), ?)",
                   (thread_id, thread_id, json.dumps(record, default=str)))

    def audit(self, thread_id: str) -> list[dict]:
        rows = self._exec("SELECT record FROM audit WHERE thread_id = ? ORDER BY seq", (thread_id,)).fetchall()
        return [json.loads(r["record"]) for r in rows]

    def expire_pending(self, now: float) -> int:
        """Proposals nobody decided on in time. Their evidence is stale, so
        approving them later would act on an old picture of the incident."""
        return self._exec("UPDATE runs SET status = 'expired', updated_at = ? "
                          "WHERE status = 'awaiting_approval' AND expires_at < ?", (now, now)).rowcount

    def mark_stale(self, now: float, lease_s: float) -> int:
        """Runs whose process stopped updating them: marked interrupted, so a
        human can recover them. A running run writes on every budget charge,
        so silence longer than the lease means nobody is running it."""
        return self._exec(f"UPDATE runs SET status = 'interrupted', updated_at = ? "
                          f"WHERE status IN {RUNNING} AND updated_at < ?", (now, now - lease_s)).rowcount

    def purge(self, before: float) -> list[str]:
        """Delete settled runs last updated before `before`. Returns their ids
        (the caller deletes their graph checkpoints too)."""
        ids = [r["thread_id"] for r in self._exec(
            f"SELECT thread_id FROM runs WHERE status IN {FINAL} AND updated_at < ?", (before,)).fetchall()]
        for thread_id in ids:
            self._exec("DELETE FROM audit WHERE thread_id = ?", (thread_id,))
            self._exec("DELETE FROM runs WHERE thread_id = ?", (thread_id,))
        return ids

    def close(self) -> None:
        self._conn.close()
