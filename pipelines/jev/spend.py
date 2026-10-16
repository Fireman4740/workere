"""One atomic paid budget across experiment clients and processes on the same host."""
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pipelines.jev.client import BudgetExceeded


class SharedBudget:
    def __init__(self, path, scope, limit_usd, initial_spend_usd=0):
        if not scope or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                            for v in (limit_usd, initial_spend_usd)) or initial_spend_usd > limit_usd:
            raise ValueError("invalid shared budget")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.scope, self.lock = scope, threading.RLock()
        self.db = sqlite3.connect(path, timeout=.25, check_same_thread=False, isolation_level=None)
        def initialize():
            self.db.execute("CREATE TABLE IF NOT EXISTS accounts(scope TEXT PRIMARY KEY, cap REAL, used REAL, initial REAL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY, scope TEXT, amount REAL, settled INTEGER)")
            self.db.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?)", (scope, limit_usd, initial_spend_usd, initial_spend_usd))
            cap, used, initial = self.db.execute("SELECT cap,used,initial FROM accounts WHERE scope=?", (scope,)).fetchone()
            if abs(cap-limit_usd) > 1e-9 or abs(initial-initial_spend_usd) > 1e-9:
                raise ValueError("shared budget identity differs; refusing to reset or increase it")
        try:
            self._transaction(initialize)
        except BaseException:
            self.db.close()
            raise

    def _transaction(self, operation):
        """Retry only the local transaction; no model call occurs inside this loop.

        A failed BEGIN used to escape after SQLite's busy timeout. Multiple
        processes can starve one writer even while the other jobs make progress.
        Roll back a failed attempt and yield between attempts. The cap and all
        existing reservations remain in the same database throughout.
        """
        deadline = time.monotonic() + 300
        retries, last_log = 0, time.monotonic()
        with self.lock:
            while True:
                try:
                    self.db.execute("BEGIN IMMEDIATE")
                    result = operation()
                    self.db.commit()
                    return result
                except BaseException as exc:
                    self.db.rollback()
                    code = getattr(exc, "sqlite_errorcode", None)
                    busy = isinstance(exc, sqlite3.OperationalError) and (
                        code is not None and code & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        or code is None and ("locked" in str(exc).lower() or "busy" in str(exc).lower()))
                    if not busy or time.monotonic() >= deadline:
                        raise
                    retries += 1
                    if time.monotonic() - last_log >= 30:
                        print(f"PLAN_BUDGET_LOCK_WAIT scope={self.scope} retries={retries}", flush=True)
                        last_log = time.monotonic()
                    # Independent jitter avoids re-synchronizing the experiment processes.
                    time.sleep(min(.01 * 2 ** min(retries, 4), .2) + (uuid.uuid4().int % 40) / 1000)

    def reserve(self, amount):
        ticket = uuid.uuid4().hex
        def operation():
            cap, used = self.db.execute("SELECT cap,used FROM accounts WHERE scope=?", (self.scope,)).fetchone()
            if used + amount > cap + 1e-12:
                raise BudgetExceeded(f"global plan budget exhausted: accounted/reserved {used:.6f} + {amount:.6f} > {cap:.2f} USD")
            self.db.execute("INSERT INTO reservations VALUES (?,?,?,0)", (ticket, self.scope, amount))
            self.db.execute("UPDATE accounts SET used=used+? WHERE scope=?", (amount, self.scope))
            return ticket
        return self._transaction(operation)

    def settle(self, ticket, cost):
        # Missing provider cost retains the full reservation, including after interruption.
        if cost is None:
            return
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid shared-budget cost")
        def operation():
            amount, settled = self.db.execute("SELECT amount,settled FROM reservations WHERE id=? AND scope=?", (ticket, self.scope)).fetchone()
            if not settled:
                self.db.execute("UPDATE accounts SET used=used+? WHERE scope=?", (cost-amount, self.scope))
                self.db.execute("UPDATE reservations SET amount=?,settled=1 WHERE id=?", (cost, ticket))
            return amount, settled
        amount, settled = self._transaction(operation)
        if not settled and cost > amount + 1e-12:
            raise BudgetExceeded("provider charge exceeded the configured conservative reservation")

    def status(self):
        with self.lock:
            cap, used, initial = self.db.execute("SELECT cap,used,initial FROM accounts WHERE scope=?", (self.scope,)).fetchone()
        return {"scope": self.scope, "limit_usd": cap, "accounted_or_reserved_usd": used,
                "initial_accounted_or_reserved_usd": initial, "remaining_usd": max(0, cap-used)}

    def close(self):
        self.db.close()
