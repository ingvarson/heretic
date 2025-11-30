# SQLite Session Persistence Implementation Plan

## Overview

Add persistent storage for Optuna optimization sessions using SQLite. This enables:
1. **Session recovery** - Resume after SSH disconnect or crash
2. **Session continuation** - Add more trials to existing sessions
3. **Session management** - List, inspect, and compare past sessions

## New CLI Commands

```bash
# List all saved sessions
heretic --list-sessions

# Continue an existing session with more trials
heretic --continue-session=<session_id> --n-trials=200

# Delete a session
heretic --delete-session=<session_id>

# Normal run (creates new session automatically)
heretic --model=... --n-trials=400
```

## Database Schema

### File Location
```
~/.heretic/sessions.db
```

### Tables

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                    -- UUID or timestamp-based ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL,                    -- Model ID/path
    status TEXT DEFAULT 'running',          -- running, completed, interrupted

    -- Settings snapshot (JSON)
    settings_json TEXT NOT NULL,

    -- Precomputed data paths (for session continuation)
    directions_path TEXT,                   -- Path to saved direction tensors

    -- Summary stats
    total_trials INTEGER DEFAULT 0,
    best_kl_divergence REAL,
    best_refusals INTEGER,

    -- Optuna study name for JournalStorage
    optuna_study_name TEXT NOT NULL
);

-- Trials table (mirrors Optuna but with extra context)
CREATE TABLE trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    trial_number INTEGER NOT NULL,

    -- Parameters (JSON for flexibility)
    parameters_json TEXT NOT NULL,

    -- Results
    kl_divergence REAL,
    refusals INTEGER,
    score_kl REAL,
    score_refusals REAL,

    -- Status
    status TEXT DEFAULT 'completed',        -- completed, pruned, failed
    duration_seconds REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(session_id, trial_number)
);

-- Index for fast lookups
CREATE INDEX idx_trials_session ON trials(session_id);
CREATE INDEX idx_sessions_model ON sessions(model);
```

## Implementation Steps

### Phase 1: Core Storage (config.py, storage.py)

#### 1.1 New file: `src/heretic/storage.py`

```python
import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .config import Settings


class SessionStorage:
    """SQLite-based session storage for Optuna optimization."""

    DB_PATH = Path.home() / ".heretic" / "sessions.db"
    DIRECTIONS_DIR = Path.home() / ".heretic" / "directions"

    def __init__(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (...);
                CREATE TABLE IF NOT EXISTS trials (...);
                -- indexes
            """)

    def create_session(
        self,
        model: str,
        settings: Settings,
        direction_variants: dict,
    ) -> str:
        """Create new session and return session ID."""
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"

        # Save direction tensors
        directions_path = self.DIRECTIONS_DIR / f"{session_id}_directions.pt"
        torch.save(direction_variants, directions_path)

        # Insert session record
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                INSERT INTO sessions (id, model, settings_json, directions_path, optuna_study_name)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                model,
                settings.model_dump_json(),
                str(directions_path),
                f"heretic_{session_id}",
            ))

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Load session by ID."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, created_at, model, status, total_trials,
                       best_kl_divergence, best_refusals
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]

    def save_trial(
        self,
        session_id: str,
        trial_number: int,
        parameters: dict,
        kl_divergence: float,
        refusals: int,
        duration: float,
        status: str = "completed",
    ):
        """Save trial result."""
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                INSERT INTO trials
                (session_id, trial_number, parameters_json, kl_divergence,
                 refusals, status, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                trial_number,
                json.dumps(parameters),
                kl_divergence,
                refusals,
                status,
                duration,
            ))

            # Update session stats
            conn.execute("""
                UPDATE sessions SET
                    total_trials = total_trials + 1,
                    updated_at = CURRENT_TIMESTAMP,
                    best_kl_divergence = MIN(COALESCE(best_kl_divergence, 999), ?),
                    best_refusals = MIN(COALESCE(best_refusals, 999), ?)
                WHERE id = ?
            """, (kl_divergence, refusals, session_id))

    def load_directions(self, session_id: str) -> dict:
        """Load precomputed direction tensors for session continuation."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        directions_path = Path(session["directions_path"])
        if not directions_path.exists():
            raise FileNotFoundError(f"Directions file missing: {directions_path}")

        return torch.load(directions_path)

    def get_optuna_storage_url(self, session_id: str) -> str:
        """Get Optuna JournalStorage path for session."""
        journal_path = self.DB_PATH.parent / f"optuna_{session_id}.log"
        return f"sqlite:///{self.DB_PATH}"  # Or use JournalStorage

    def delete_session(self, session_id: str):
        """Delete session and associated files."""
        session = self.get_session(session_id)
        if session:
            # Delete directions file
            directions_path = Path(session["directions_path"])
            if directions_path.exists():
                directions_path.unlink()

            # Delete from DB
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.execute("DELETE FROM trials WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
```

#### 1.2 Config additions (`config.py`)

```python
# New CLI options
continue_session: str | None = Field(
    default=None,
    description="Continue an existing session by ID.",
)

list_sessions: bool = Field(
    default=False,
    description="List all saved sessions and exit.",
)

delete_session: str | None = Field(
    default=None,
    description="Delete a session by ID and exit.",
)

disable_persistence: bool = Field(
    default=False,
    description="Run without saving session to database.",
)
```

### Phase 2: Main Integration (main.py)

#### 2.1 Session handling in `run()`

```python
def run():
    # ... existing setup ...

    # Handle session management commands
    if settings.list_sessions:
        storage = SessionStorage()
        sessions = storage.list_sessions()
        # Print formatted table
        return

    if settings.delete_session:
        storage = SessionStorage()
        storage.delete_session(settings.delete_session)
        print(f"Deleted session: {settings.delete_session}")
        return

    # Session continuation mode
    if settings.continue_session:
        storage = SessionStorage()
        session = storage.get_session(settings.continue_session)
        if not session:
            print(f"[red]Session not found: {settings.continue_session}[/]")
            return

        # Load saved settings and directions
        saved_settings = Settings.model_validate_json(session["settings_json"])
        direction_variants = storage.load_directions(settings.continue_session)

        # Load existing Optuna study
        study = optuna.load_study(
            study_name=session["optuna_study_name"],
            storage=storage.get_optuna_storage_url(settings.continue_session),
        )

        # Continue optimization
        # ...
    else:
        # Normal new session flow
        # ... existing code to compute directions ...

        if not settings.disable_persistence:
            storage = SessionStorage()
            session_id = storage.create_session(
                model=settings.model,
                settings=settings,
                direction_variants=direction_variants,
            )
            print(f"Session ID: [bold]{session_id}[/]")

        # Create Optuna study with storage
        study = optuna.create_study(
            study_name=f"heretic_{session_id}",
            storage=storage.get_optuna_storage_url(session_id),
            # ...
        )
```

#### 2.2 Trial saving in objective function

```python
def objective(trial: Trial) -> tuple[float, float]:
    # ... existing trial code ...

    # Save trial to database
    if not settings.disable_persistence:
        storage.save_trial(
            session_id=session_id,
            trial_number=trial_index,
            parameters=get_trial_parameters(trial),
            kl_divergence=kl_divergence,
            refusals=refusals,
            duration=trial_duration,
        )

    return score
```

### Phase 3: CLI Output Formatting

#### 3.1 `--list-sessions` output

```
Heretic Sessions
================

ID                          Model                              Trials  Best KL  Best Ref  Status       Created
--------------------------  ---------------------------------  ------  -------  --------  -----------  -------------------
20241130_143022_a1b2c3d4    Qwen/Qwen3-Next-80B-A3B-Instruct     400     0.23        3    completed   2024-11-30 14:30:22
20241129_091512_e5f6g7h8    meta-llama/Llama-3.1-70B-Instruct    200     0.45        8    completed   2024-11-29 09:15:12
20241128_220145_i9j0k1l2    Qwen/Qwen3-4B-Instruct                50     0.12        5    interrupted 2024-11-28 22:01:45

Use --continue-session=<ID> to add more trials to a session.
```

### Phase 4: Optuna Storage Backend

Use Optuna's built-in RDB storage for full study persistence:

```python
from optuna.storages import RDBStorage

storage = RDBStorage(
    url=f"sqlite:///{Path.home()}/.heretic/sessions.db",
    engine_kwargs={"connect_args": {"timeout": 30}},
)

study = optuna.create_study(
    study_name=f"heretic_{session_id}",
    storage=storage,
    load_if_exists=True,  # For continuation
    directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
)
```

## File Structure

```
~/.heretic/
├── sessions.db                    # SQLite database
├── directions/
│   ├── 20241130_143022_a1b2c3d4_directions.pt
│   └── 20241129_091512_e5f6g7h8_directions.pt
└── optuna/                        # Optional: separate Optuna journals
    └── ...
```

## Implementation Checklist

### Phase 1: Core Storage
- [ ] Create `src/heretic/storage.py` with SessionStorage class
- [ ] Add database schema initialization
- [ ] Implement session CRUD operations
- [ ] Implement trial saving
- [ ] Add direction tensor save/load

### Phase 2: Config & CLI
- [ ] Add new CLI options to `config.py`
- [ ] Add `--list-sessions` command
- [ ] Add `--continue-session` command
- [ ] Add `--delete-session` command
- [ ] Add `--disable-persistence` flag

### Phase 3: Main Integration
- [ ] Modify `run()` to handle session commands
- [ ] Create session on new runs
- [ ] Save trials during optimization
- [ ] Load session for continuation
- [ ] Handle interrupted sessions gracefully

### Phase 4: Optuna Integration
- [ ] Use RDBStorage for Optuna studies
- [ ] Enable `load_if_exists` for continuation
- [ ] Preserve sampler state across sessions

### Phase 5: Testing
- [ ] Test new session creation
- [ ] Test session continuation
- [ ] Test session listing
- [ ] Test recovery after disconnect
- [ ] Test with 70B+ models

## Notes

1. **Direction tensors must be saved** - They're computed from residuals which are deleted after computation. Without saving, session continuation requires re-computing residuals.

2. **Optuna RDBStorage vs JournalStorage** - RDBStorage integrates better with SQLite and provides full study persistence. JournalStorage is simpler but separate.

3. **Session ID format** - `YYYYMMDD_HHMMSS_<random>` provides sortability and uniqueness.

4. **Backwards compatibility** - The `--disable-persistence` flag allows running without database for quick tests.
