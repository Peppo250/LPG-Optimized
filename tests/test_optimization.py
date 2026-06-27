import unittest
from datetime import datetime, timedelta
from lpg_catering.optimization.caterer import (
    CatererProfile,
    EventDetails,
    CatererOptimizer,
    optimize_single_event
)
from lpg_catering.optimization.regional import RegionalOptimizer
from lpg_catering.optimization.lp_solver import lp_optimize_procurement

class TestCatererOptimizer(unittest.TestCase):
    def setUp(self):
        self.profile_expert = CatererProfile(
            caterer_id="CAT001",
            name="Expert Murugan",
            business_size="large",
            experience_yrs=15,
            num_burners=12,
            base_region="Chennai"
        )
        self.profile_novice = CatererProfile(
            caterer_id="CAT002",
            name="Novice Balaji",
            business_size="small",
            experience_yrs=1,
            num_burners=3,
            base_region="default"
        )

    def test_predict_consumption_bounds(self):
        event = EventDetails(
            event_date="2026-11-15",
            event_type="wedding",
            headcount=500,
            num_dishes=7,
            duration_hrs=6.0,
            menu_profile="mixed_standard",
            is_festival_season=True
        )
        
        optimizer = CatererOptimizer(self.profile_expert)
        pred = optimizer.predict_consumption(event)
        
        self.assertTrue(pred > 0)
        self.assertTrue(pred < 500)  # Reasonable upper bound

    def test_calculate_cylinders_buffer(self):
        optimizer_novice = CatererOptimizer(self.profile_novice)
        optimizer_expert = CatererOptimizer(self.profile_expert)
        
        # Novice (exp < 3) should have 20% safety buffer
        cyl_novice = optimizer_novice.calculate_cylinders(100.0)  # 100 * 1.2 / 17.5 = ceil(6.85) = 7
        self.assertEqual(cyl_novice, 7)
        
        # Expert (exp >= 6) should have 10% safety buffer
        cyl_expert = optimizer_expert.calculate_cylinders(100.0)  # 100 * 1.1 / 17.5 = ceil(6.28) = 7
        self.assertEqual(cyl_expert, 7)

    def test_calculate_order_timing(self):
        optimizer = CatererOptimizer(self.profile_expert)
        event_date = "2026-11-15"
        
        # High risk, expert in Chennai (lead time = 1 day)
        # Base buffer = 1 + 1 = 2 days
        # High risk (>50) = 3 days risk buffer
        # Expected offset = 2 + 3 = 5 days recommended
        rec, latest = optimizer.calculate_order_timing(event_date, 60.0)
        
        self.assertEqual(rec, "2026-11-10")
        self.assertEqual(latest, "2026-11-13")

    def test_efficiency_score(self):
        optimizer = CatererOptimizer(self.profile_expert)
        # Ideal order: consumption matches usable capacity
        eff = optimizer.efficiency_score(35.0, 3.5, 2)  # usable = 35.0, wastage = 3.5, over_order = 0
        self.assertTrue(eff >= 50)
        self.assertTrue(eff <= 100)


class TestRegionalOptimizer(unittest.TestCase):
    def test_demand_smoothing(self):
        optimizer = RegionalOptimizer(dealer_daily_capacity_cylinders=10)
        
        # Create mock results
        from lpg_catering.optimization.caterer import OptimizationResult
        results = [
            OptimizationResult(
                caterer_id="CAT001",
                event_date="2026-11-15",
                predicted_consumption_kg=70.0,
                cylinders_to_order=5,
                recommended_order_date="2026-11-10",
                latest_safe_order_date="2026-11-13",
                stockout_risk_pct=10.0,
                wastage_estimate_kg=5.0,
                estimated_cost_inr=9550.0,
                efficiency_score=85.0,
                recommendation_tier="GREEN",
                action_items=[]
            ),
            OptimizationResult(
                caterer_id="CAT002",
                event_date="2026-11-15",
                predicted_consumption_kg=120.0,
                cylinders_to_order=8,
                recommended_order_date="2026-11-10",
                latest_safe_order_date="2026-11-12",
                stockout_risk_pct=5.0,
                wastage_estimate_kg=8.0,
                estimated_cost_inr=15280.0,
                efficiency_score=80.0,
                recommendation_tier="GREEN",
                action_items=[]
            )
        ]
        
        # Today base test for rolling window
        df = optimizer.build_demand_curve(results, days_window=30)
        self.assertEqual(len(df), 30)
        
        # Test smoothing shifts
        smoothed_results, df_smoothed = optimizer.smooth_demand(results, df)
        self.assertEqual(len(smoothed_results), len(results))


class TestLPSolver(unittest.TestCase):
    def test_lp_optimize_basic(self):
        events = [
            {"event_id": "EV1", "event_date": "2026-11-15", "consumption_kg": 30.0},
            {"event_id": "EV2", "event_date": "2026-11-16", "consumption_kg": 50.0}
        ]
        
        res = lp_optimize_procurement(events, budget_inr=50000.0)
        self.assertIn(res["status"], ["optimal", "fallback"])
        self.assertTrue(len(res["allocations"]) > 0)

if __name__ == "__main__":
    unittest.main()
