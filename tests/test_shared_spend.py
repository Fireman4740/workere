import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipelines.jev.client import BudgetExceeded
from pipelines.jev.spend import SharedBudget


class SharedSpend(unittest.TestCase):
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
