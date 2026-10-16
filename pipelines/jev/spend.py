"""One atomic paid budget across experiment clients and processes on the same host."""
import math
import sqlite3
import threading
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
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS accounts(scope TEXT PRIMARY KEY, cap REAL, used REAL, initial REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY, scope TEXT, amount REAL, settled INTEGER)")
        self.db.commit()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT OR IGNORE INTO accounts VALUES (?,?,?,?)", (scope, limit_usd, initial_spend_usd, initial_spend_usd))
                cap, used, initial = self.db.execute("SELECT cap,used,initial FROM accounts WHERE scope=?", (scope,)).fetchone()
                if abs(cap-limit_usd) > 1e-9 or abs(initial-initial_spend_usd) > 1e-9:
                    raise ValueError("shared budget identity differs; refusing to reset or increase it")
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def reserve(self, amount):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                cap, used = self.db.execute("SELECT cap,used FROM accounts WHERE scope=?", (self.scope,)).fetchone()
                if used + amount > cap + 1e-12:
                    raise BudgetExceeded(f"global plan budget exhausted: accounted/reserved {used:.6f} + {amount:.6f} > {cap:.2f} USD")
                ticket = uuid.uuid4().hex
                self.db.execute("INSERT INTO reservations VALUES (?,?,?,0)", (ticket, self.scope, amount))
                self.db.execute("UPDATE accounts SET used=used+? WHERE scope=?", (amount, self.scope))
                self.db.commit()
                return ticket
            except BaseException:
                self.db.rollback()
                raise

    def settle(self, ticket, cost):
        # Missing provider cost retains the full reservation, including after interruption.
        if cost is None:
            return
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid shared-budget cost")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                amount, settled = self.db.execute("SELECT amount,settled FROM reservations WHERE id=? AND scope=?", (ticket, self.scope)).fetchone()
                if not settled:
                    self.db.execute("UPDATE accounts SET used=used+? WHERE scope=?", (cost-amount, self.scope))
                    self.db.execute("UPDATE reservations SET amount=?,settled=1 WHERE id=?", (cost, ticket))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
        if not settled and cost > amount + 1e-12:
            raise BudgetExceeded("provider charge exceeded the configured conservative reservation")

    def status(self):
        with self.lock:
            cap, used, initial = self.db.execute("SELECT cap,used,initial FROM accounts WHERE scope=?", (self.scope,)).fetchone()
        return {"scope": self.scope, "limit_usd": cap, "accounted_or_reserved_usd": used,
                "initial_accounted_or_reserved_usd": initial, "remaining_usd": max(0, cap-used)}

    def close(self):
        self.db.close()
