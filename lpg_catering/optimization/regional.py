import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from lpg_catering.optimization.caterer import OptimizationResult

class RegionalOptimizer:
    """
    Aggregates individual caterer optimizations into a regional
    demand forecast and smoothing plan for LPG dealers.
    """

    def __init__(self, region: str = "Tamil Nadu",
                 dealer_daily_capacity_cylinders: int = 200):
        self.region = region
        self.daily_cap = dealer_daily_capacity_cylinders

    def build_demand_curve(self, optimization_results: List[OptimizationResult],
                            days_window: int = 30) -> pd.DataFrame:
        """
        Aggregate all caterer order recommendations into a daily demand curve.
        Returns a DataFrame with date, raw_demand, smoothed_demand, capacity_pressure.
        """
        today = datetime.today()
        dates = [today + timedelta(days=i) for i in range(days_window)]
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]

        demand = {d: 0 for d in date_strs}

        for r in optimization_results:
            if r.recommended_order_date in demand:
                demand[r.recommended_order_date] += r.cylinders_to_order

        df = pd.DataFrame({
            "date":          date_strs,
            "raw_demand":    [demand[d] for d in date_strs],
            "is_weekend":    [datetime.strptime(d, "%Y-%m-%d").weekday() >= 5 for d in date_strs],
        })
        df["smoothed_demand"] = df["raw_demand"].rolling(3, center=True, min_periods=1).mean().round(1)
        df["capacity_pressure"] = (df["raw_demand"] / self.daily_cap * 100).round(1)
        df["spike_flag"] = df["raw_demand"] > self.daily_cap * 0.80
        return df

    def smooth_demand(self, optimization_results: List[OptimizationResult],
                       demand_df: pd.DataFrame) -> Tuple[List[OptimizationResult], pd.DataFrame]:
        """
        Demand smoothing: shift caterer order dates to flatten spikes.

        Algorithm:
          1. Sort days by capacity pressure (highest first)
          2. For spike days, shift some orders 1-2 days earlier
          3. Prioritise shifting low-stockout-risk caterers (they have margin)
          4. Never shift past a caterer's latest_safe_order_date

        Returns adjusted results and updated demand curve.
        """
        adjusted = list(optimization_results)
        demand_counts = demand_df.set_index("date")["raw_demand"].to_dict()

        spike_dates = demand_df[demand_df["spike_flag"]]["date"].tolist()

        for spike_date in spike_dates:
            # Sort caterers on this day by risk ascending (lowest risk → safest to shift)
            on_spike = [(i, r) for i, r in enumerate(adjusted)
                        if r.recommended_order_date == spike_date]
            on_spike.sort(key=lambda x: x[1].stockout_risk_pct)

            # Shift up to half of them 1-2 days earlier
            n_to_shift = len(on_spike) // 2
            for idx, (list_idx, result) in enumerate(on_spike[:n_to_shift]):
                shift_days = 2 if result.stockout_risk_pct < 15 else 1
                orig = datetime.strptime(result.recommended_order_date, "%Y-%m-%d")
                latest = datetime.strptime(result.latest_safe_order_date, "%Y-%m-%d")
                new_date = orig - timedelta(days=shift_days)

                # Only shift if still before latest safe date
                if new_date <= latest:
                    new_date_str = new_date.strftime("%Y-%m-%d")
                    # Update demand
                    if spike_date in demand_counts:
                        demand_counts[spike_date] = max(0, demand_counts[spike_date] - result.cylinders_to_order)
                    demand_counts[new_date_str] = demand_counts.get(new_date_str, 0) + result.cylinders_to_order
                    # Update result
                    from copy import deepcopy
                    new_result = deepcopy(result)
                    new_result.recommended_order_date = new_date_str
                    new_result.optimization_notes += f" | Shifted {shift_days}d earlier (demand smoothing)"
                    adjusted[list_idx] = new_result

        # Rebuild demand df
        demand_df["smoothed_orders"] = demand_df["date"].map(demand_counts).fillna(0)
        demand_df["peak_reduction_pct"] = (
            (demand_df["raw_demand"] - demand_df["smoothed_orders"])
            / demand_df["raw_demand"].replace(0, 1) * 100
        ).round(1)

        return adjusted, demand_df

    def regional_summary(self, demand_df: pd.DataFrame,
                          optimization_results: List[OptimizationResult]) -> Dict:
        """Generate a regional KPI summary for the dealer dashboard."""
        total_cylinders = sum(r.cylinders_to_order for r in optimization_results)
        total_cost      = sum(r.estimated_cost_inr for r in optimization_results)
        avg_stockout    = np.mean([r.stockout_risk_pct for r in optimization_results])
        spike_days      = int(demand_df["spike_flag"].sum())
        peak_demand     = int(demand_df["raw_demand"].max())
        avg_daily       = round(float(demand_df["raw_demand"].mean()), 1)

        high_risk = [r for r in optimization_results if r.recommendation_tier == "RED"]
        return {
            "region":                   self.region,
            "total_caterers":           len(optimization_results),
            "total_cylinders_30d":      total_cylinders,
            "total_revenue_inr":        total_cost,
            "avg_stockout_risk_pct":    round(avg_stockout, 1),
            "peak_demand_cylinders":    peak_demand,
            "avg_daily_demand":         avg_daily,
            "dealer_capacity":          self.daily_cap,
            "spike_days_count":         spike_days,
            "high_risk_caterers":       len(high_risk),
            "capacity_utilization_pct": round(avg_daily / self.daily_cap * 100, 1),
        }
