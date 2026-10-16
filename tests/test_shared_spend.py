import tempfile
import sqlite3
import multiprocessing
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipelines.jev.client import BudgetExceeded, DecisionClient
from pipelines.jev.spend import SharedBudget


def _hold_external_writer(path, ready, release):
    db = sqlite3.connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        ready.set()
        release.wait(5)
        db.rollback()
    finally:
        db.close()


class SharedSpend(unittest.TestCase):
    def _hold_writer(self, path):
        ctx = multiprocessing.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        thread = ctx.Process(target=_hold_external_writer, args=(path, ready, release))
        thread.start()
        self.assertTrue(ready.wait(2))
        return thread, release

    def test_reservation_waits_through_another_process_writer_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "budget.sqlite")
            budget = SharedBudget(path, "plan", 1, .25)
            budget.db.execute("PRAGMA busy_timeout=10")
            thread, release = self._hold_writer(path)
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(budget.reserve, .1)
                time.sleep(.06)
                self.assertFalse(future.done(), "a temporary ledger lock must not fail the run")
                release.set()
                ticket = future.result(timeout=3)
                self.assertAlmostEqual(budget.status()["accounted_or_reserved_usd"], .35)
                budget.settle(ticket, .01)
                self.assertAlmostEqual(budget.status()["accounted_or_reserved_usd"], .26)
            finally:
                release.set(); thread.join(); pool.shutdown(); budget.close()

    def test_settlement_lock_does_not_repeat_a_paid_model_call(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "budget.sqlite")
            holder, timer = [], []
            calls = []
            def transport(body):
                calls.append(body)
                thread, release = self._hold_writer(path)
                holder.append((thread, release))
                timer.append(threading.Timer(.12, release.set))
                timer[-1].start()
                return {"model": "paid-test", "answers": {"causal": {"type": "noul", "noul": .8}}, "usage": {"cost": .01}}
            with DecisionClient(model="paid-test", model_version="test", mode="live", transport=transport,
                    cache_path=str(Path(temp) / "cache.sqlite"), input_price_per_million=0,
                    budget_usd=1, max_cost_per_call_usd=.02, requests_per_minute=6000,
                    shared_budget={"path": path, "scope": "plan", "limit_usd": 1}) as client:
                client.shared_budget.db.execute("PRAGMA busy_timeout=10")
                questions = {"causal": {"type": "noul", "instructions": "causal?", "criteria": {"true": "yes", "false": "no"}}}
                try:
                    first = client.decide("state", questions)
                    second = client.decide("state", questions)
                    self.assertEqual(first["response"]["answers"]["causal"]["noul"], .8)
                    self.assertTrue(second["cache_hit"])
                    self.assertEqual(len(calls), 1)
                    self.assertAlmostEqual(client.shared_budget.status()["accounted_or_reserved_usd"], .01)
                finally:
                    for thread, release in holder:
                        release.set(); thread.join()
                    for t in timer:
                        t.join()

    def test_parallel_clients_cannot_oversubscribe_and_resume_keeps_reservations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "budget.sqlite")
            clients = [SharedBudget(path, "plan", 1, .25) for _ in range(4)]
            def reserve(index):
                try:
                    return clients[index % 4].reserve(.125)
                except BudgetExceeded:
                    return None
            with ThreadPoolExecutor(max_workers=32) as pool:
                tickets = [t for t in pool.map(reserve, range(64)) if t]
            self.assertEqual(len(tickets), 6)
            self.assertAlmostEqual(clients[0].status()["accounted_or_reserved_usd"], 1)
            clients[0].settle(tickets[0], .025)
            clients[0].settle(tickets[0], .025)  # settlement is idempotent
            clients[0].settle(tickets[1], None)  # unknown cost cannot free the reservation
            self.assertAlmostEqual(clients[0].status()["accounted_or_reserved_usd"], .9)
            for client in clients:
                client.close()
            resumed = SharedBudget(path, "plan", 1, .25)
            self.assertAlmostEqual(resumed.status()["accounted_or_reserved_usd"], .9)
            with self.assertRaises(BudgetExceeded):
                resumed.reserve(.125)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
