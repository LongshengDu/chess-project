"""Transactional storage for studies, cached analysis and local play sessions."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time
import uuid

from analysis.accuracy.figures import DEFAULT_FIGURE_NAMES
from backend.settings import CONFIG


class GameRepository:
    """One connection per operation; no model work is performed in transactions."""

    def __init__(self, database: Path):
        self.database = Path(database)
        self.output_directory = Path(CONFIG['SERVER']['STORAGE']['OUTPUT_DIR'])
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, snapshot TEXT NOT NULL,
                    favorite INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY REFERENCES games(id) ON DELETE CASCADE,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_play (
                    id TEXT PRIMARY KEY REFERENCES games(id) ON DELETE CASCADE,
                    request_id TEXT UNIQUE, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS full_analyses (
                    id TEXT PRIMARY KEY REFERENCES games(id) ON DELETE CASCADE,
                    data TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=CONFIG['SERVER']['STORAGE']['LOCK_TIMEOUT_SECONDS'])
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, snapshot, name):
        game_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO games(id,name,snapshot,created) VALUES(?,?,?,?)',
                       (game_id, name, json.dumps(snapshot), time.time()))
        return game_id

    def latest_id(self):
        with self.connect() as db:
            row = db.execute('SELECT id FROM games ORDER BY created DESC LIMIT 1').fetchone()
        return row['id'] if row else None

    def list_games(self, kind, page):
        # Preserve the upstream custom view, which also contains local play games.
        condition = {'custom': '', 'favorites': 'WHERE g.favorite != 0',
                     'play': 'WHERE p.id IS NOT NULL'}.get(kind, 'WHERE 0')
        source = f'FROM games g LEFT JOIN local_play p ON p.id = g.id {condition}'
        with self.connect() as db:
            db.execute('BEGIN')  # Counts and rows describe the same read snapshot.
            total = db.execute(f'SELECT count(*) {source}').fetchone()[0]
            rows = db.execute(f'SELECT g.*, p.state {source} ORDER BY g.created DESC LIMIT 25 OFFSET ?',
                              ((page - 1) * 25,)).fetchall()
        games = []
        for row in rows:
            state = json.loads(row['state']) if row['state'] else None
            game = {'game_id': row['id'], 'game_type': 'play' if state else 'custom',
                    'custom_name': row['name'], 'is_favorited': bool(row['favorite']),
                    'result': json.loads(row['snapshot'])['result']}
            if state:
                game.update(maia_name=f"maia_kdd_{state['maia_rating']}", player_color=state['player_color'])
            games.append(game)
        return {'total_games': total, 'total_pages': max(1, (total + 24) // 25), 'games': games}

    def get(self, game_id):
        with self.connect() as db:
            row = db.execute('SELECT name,snapshot FROM games WHERE id=?', (game_id,)).fetchone()
        return None if row is None else {'game_id': game_id, 'name': row['name'],
                                        'snapshot': json.loads(row['snapshot'])}

    def update_metadata(self, game_id, data):
        with self.connect() as db:
            if db.execute('SELECT id FROM games WHERE id=?', (game_id,)).fetchone() is None:
                return False
            if 'custom_name' in data:
                db.execute('UPDATE games SET name=? WHERE id=?', (str(data['custom_name'])[:200], game_id))
            if 'is_favorited' in data:
                db.execute('UPDATE games SET favorite=? WHERE id=?', (bool(data['is_favorited']), game_id))
        return True

    def delete(self, game_id):
        with self.connect() as db:
            db.execute('DELETE FROM games WHERE id=?', (game_id,))

    def save_analysis(self, game_id, positions):
        with self.connect() as db:
            # Test existence and write together: deletion cannot interleave and
            # turn a missing study into a foreign-key error.
            result = db.execute('INSERT OR REPLACE INTO analyses(id,data) '
                                'SELECT id,? FROM games WHERE id=?', (json.dumps(positions), game_id))
            return result.rowcount == 1

    def load_analysis(self, game_id):
        with self.connect() as db:
            row = db.execute('SELECT data FROM analyses WHERE id=?', (game_id,)).fetchone()
        return json.loads(row['data']) if row else []

    def accuracy_output_directory(self, game_id):
        """Stable user-output destination, independent of the analysis cache."""
        if not isinstance(game_id, str) or len(game_id) != 32 or any(c not in '0123456789abcdef' for c in game_id):
            raise ValueError('Invalid saved game identifier for accuracy output.')
        return self.output_directory / f'{game_id}-full'

    @contextmanager
    def _publish_accuracy_figures(self, game_id, source):
        """Replace only owned artifacts; restore them if publication fails."""
        names = DEFAULT_FIGURE_NAMES
        if any(not (source / name).is_file() for name in names):
            raise ValueError('All accuracy figures must be staged before publication.')
        output = self.accuracy_output_directory(game_id)
        existed = output.exists()
        output.mkdir(parents=True, exist_ok=True)
        moved, installed = [], []
        with TemporaryDirectory(prefix='.previous-accuracy-', dir=source.parent) as previous:
            backup = Path(previous)
            try:
                for name in names:
                    target = output / name
                    if target.exists():
                        target.replace(backup / name)
                        moved.append(name)
                    replacement = source / name
                    replacement.replace(target)
                    installed.append(name)
                yield
            except BaseException:
                for name in installed:
                    (output / name).unlink(missing_ok=True)
                for name in moved:
                    (output / name).parent.mkdir(parents=True, exist_ok=True)
                    (backup / name).replace(output / name)
                if not existed and not any(output.iterdir()):
                    output.rmdir()
                raise

    def save_full_analysis(self, game_id, analysis, *, accuracy_figures=None):
        """Publish evidence, UI positions and optional staged accuracy figures.

        Rendering happens before this transaction. The write transaction orders
        publication against deletion; file backups survive until its commit.
        """
        with ExitStack() as artifacts:
            with self.connect() as db:
                result = db.execute('INSERT OR REPLACE INTO full_analyses(id,data) '
                                    'SELECT id,? FROM games WHERE id=?',
                                    (json.dumps({key: value for key, value in analysis.items()
                                                 if key != 'positions'}, allow_nan=False), game_id))
                if result.rowcount != 1:
                    return False
                db.execute('INSERT OR REPLACE INTO analyses(id,data) VALUES(?,?)',
                           (game_id, json.dumps(analysis['positions'], allow_nan=False)))
                if accuracy_figures is not None:
                    artifacts.enter_context(self._publish_accuracy_figures(game_id, Path(accuracy_figures)))
        return True

    def load_full_analysis(self, game_id):
        with self.connect() as db:
            row = db.execute('SELECT data FROM full_analyses WHERE id=?', (game_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row['data'])

    def get_play(self, game_id):
        with self.connect() as db:
            row = db.execute('SELECT state FROM local_play WHERE id=?', (game_id,)).fetchone()
        return json.loads(row['state']) if row else None

    def create_play(self, state, snapshot, request_id):
        """Atomically create a game, or return the state of a retried start request."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if request_id is not None:
                previous = db.execute('SELECT state FROM local_play WHERE request_id=?', (request_id,)).fetchone()
                if previous:
                    return json.loads(previous['state'])
            db.execute('INSERT INTO games(id,name,snapshot,created) VALUES(?,?,?,?)',
                       (state['game_id'], snapshot['title'], json.dumps(snapshot), time.time()))
            db.execute('INSERT INTO local_play(id,request_id,state) VALUES(?,?,?)',
                       (state['game_id'], request_id, json.dumps(state)))
        return state

    def save_play(self, state, snapshot, *, invalidate=False):
        with self.connect() as db:
            db.execute('UPDATE local_play SET state=? WHERE id=?', (json.dumps(state), state['game_id']))
            db.execute('UPDATE games SET snapshot=? WHERE id=?', (json.dumps(snapshot), state['game_id']))
            if invalidate:
                db.execute('DELETE FROM analyses WHERE id=?', (state['game_id'],))
                db.execute('DELETE FROM full_analyses WHERE id=?', (state['game_id'],))

    def play_states(self):
        with self.connect() as db:
            return [json.loads(row['state']) for row in db.execute('SELECT state FROM local_play')]
