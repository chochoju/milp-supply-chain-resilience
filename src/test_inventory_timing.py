import unittest

import numpy as np
import pandas as pd

from optimize_inventory_milp import compute_component_fill_rate, solve_milp_subproblem


class InventoryTimingTests(unittest.TestCase):
    def solve(self, lead_weeks, carried_backlog=0):
        components = pd.DataFrame([{
            "component_id": "module", "holding_cost_wk": 1.0,
            "stockout_penalty_wk": 50.0, "unit_volume_m3": 0.1,
        }])
        lanes = pd.DataFrame([{
            "component_id": "module", "supplier_id": "supplier",
            "lot_size_units": 10, "min_lots": 1, "unit_cost": 1.0,
            "sec301_tariff_cost": 0.0, "unit_freight_cost": 0.0,
            "domestic_credit": 0.0, "fixed_order_fee": 0.0,
            "freight_mode": "std", "lead_time_wks": lead_weeks,
        }])
        return solve_milp_subproblem(
            components, lanes, np.array([0.0]), np.array([carried_backlog]),
            np.zeros((1, 1)), np.array([[10.0]]),
            {("module", "supplier", 0): 10}, 1, inventory_week_offset=1,
        )

    def test_one_week_lead_order_meets_next_week_demand(self):
        orders, _, _, backlog, _, _ = self.solve(lead_weeks=1)
        self.assertEqual(orders[0, 0], 1)
        self.assertAlmostEqual(backlog[0, 0], 0)

    def test_order_arriving_after_horizon_is_excluded(self):
        orders, _, _, backlog, _, _ = self.solve(lead_weeks=2)
        self.assertEqual(orders[0, 0], 0)
        self.assertAlmostEqual(backlog[0, 0], 10)

    def test_carried_backlog_is_included_in_starting_state(self):
        orders, _, _, backlog, _, _ = self.solve(lead_weeks=1, carried_backlog=4)
        self.assertEqual(orders[0, 0], 1)
        self.assertAlmostEqual(backlog[0, 0], 4)

    def test_flat_backlog_can_hide_new_shortages(self):
        demand = np.array([[10, 10]])
        backlog = np.array([[4, 4]])
        self.assertAlmostEqual(compute_component_fill_rate(backlog, demand), 60)


if __name__ == "__main__":
    unittest.main()
