import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from dataclasses import asdict
from lpg_catering.config import CYLINDER_PRICE_INR, GAS_PER_100_GUESTS
from lpg_catering.optimization.caterer import CatererProfile, EventDetails, CatererOptimizer
from lpg_catering.optimization.regional import RegionalOptimizer

class SimulationEngine:
    """
    Simulate optimization across N caterers over 30 days.
    Produces before/after metrics to demonstrate impact.
    """

    def __init__(self, n_caterers: int = 50, seed: int = 42):
        self.n_caterers = n_caterers
        self.seed = seed
        np.random.seed(seed)

    def generate_caterer_fleet(self) -> List[CatererProfile]:
        """Generate a realistic fleet of caterers."""
        sizes = np.random.choice(["small", "medium", "large"],
                                  self.n_caterers, p=[0.4, 0.45, 0.15])
        return [
            CatererProfile(
                caterer_id=f"CAT{i+1:03d}",
                name=f"Caterer {i+1}",
                business_size=sizes[i],
                experience_yrs=int(np.random.uniform(
                    *{"small":(1,5),"medium":(3,12),"large":(7,25)}[sizes[i]]
                )),
                num_burners=int(np.random.uniform(
                    *{"small":(2,4),"medium":(4,8),"large":(8,20)}[sizes[i]]
                )),
            )
            for i in range(self.n_caterers)
        ]

    def generate_events(self, caterers: List[CatererProfile],
                         month: int = 11) -> List[Tuple[CatererProfile, EventDetails]]:
        """Generate events for each caterer in a given month."""
        today = datetime.today()
        pairs = []
        for cat in caterers:
            n_events = int(np.random.uniform(2, 6))
            for _ in range(n_events):
                days_out = int(np.random.uniform(3, 30))
                event_date = (today + timedelta(days=days_out)).strftime("%Y-%m-%d")
                is_wed_season = month in [1, 2, 10, 11, 12]
                etype = np.random.choice(
                    ["wedding", "corporate_lunch", "college_canteen", "birthday_party"],
                    p=[0.45, 0.25, 0.20, 0.10] if is_wed_season else [0.15, 0.35, 0.35, 0.15]
                )
                hc_range = {"small": (50, 300), "medium": (200, 800), "large": (500, 2000)}
                hc = int(np.random.uniform(*hc_range[cat.business_size]))
                pairs.append((cat, EventDetails(
                    event_date=event_date,
                    event_type=etype,
                    headcount=hc,
                    num_dishes=int(np.random.uniform(4, 9)),
                    duration_hrs=round(np.random.uniform(3, 8), 1),
                    menu_profile=np.random.choice(list(GAS_PER_100_GUESTS.keys())),
                    is_festival_season=is_wed_season,
                )))
        return pairs

    def run(self, month: int = 11) -> Dict:
        """
        Full simulation: generate fleet → optimize → smooth → compare.
        """
        caterers = self.generate_caterer_fleet()
        events   = self.generate_events(caterers, month)

        # Before optimization: naive booking (order on day of event)
        naive_stockouts = 0
        naive_waste_kg  = 0.0
        naive_cost      = 0

        # After optimization
        opt_results = []
        for cat, evt in events:
            optimizer = CatererOptimizer(cat)
            result = optimizer.optimize_event(evt)
            opt_results.append(result)
            if result.stockout_risk_pct > 50:
                naive_stockouts += 1
            naive_waste_kg += result.wastage_estimate_kg * 1.4  # 40% more naive waste
            naive_cost     += result.cylinders_to_order * CYLINDER_PRICE_INR * 1.08

        # Regional smoothing
        reg_opt = RegionalOptimizer(dealer_daily_capacity_cylinders=300)
        demand_df = reg_opt.build_demand_curve(opt_results)
        smoothed_results, demand_df = reg_opt.smooth_demand(opt_results, demand_df)
        summary = reg_opt.regional_summary(demand_df, smoothed_results)

        # Metrics comparison
        opt_stockout_risk = np.mean([r.stockout_risk_pct for r in smoothed_results])
        opt_waste_kg      = sum(r.wastage_estimate_kg for r in smoothed_results)
        opt_cost          = sum(r.estimated_cost_inr for r in smoothed_results)
        peak_raw          = int(demand_df["raw_demand"].max())
        peak_smoothed     = int(demand_df.get("smoothed_orders", demand_df["smoothed_demand"]).max())

        return {
            "simulation": {
                "caterers":     self.n_caterers,
                "total_events": len(events),
                "month":        month,
            },
            "before_optimization": {
                "stockout_events":   naive_stockouts,
                "total_wastage_kg":  round(naive_waste_kg, 1),
                "total_cost_inr":    naive_cost,
                "peak_daily_demand": peak_raw,
            },
            "after_optimization": {
                "avg_stockout_risk_pct": round(float(opt_stockout_risk), 1),
                "total_wastage_kg":      round(opt_waste_kg, 1),
                "total_cost_inr":        int(opt_cost),
                "peak_daily_demand":     peak_smoothed,
            },
            "improvement": {
                "wastage_reduction_pct":      round((naive_waste_kg - opt_waste_kg) / max(naive_waste_kg, 1) * 100, 1),
                "cost_saving_inr":            naive_cost - int(opt_cost),
                "peak_demand_reduction_pct":  round((peak_raw - peak_smoothed) / max(peak_raw, 1) * 100, 1),
            },
            "regional_summary": summary,
            "sample_results":   [asdict(r) for r in smoothed_results[:5]],
            "demand_curve":     demand_df.to_dict("records"),
        }
