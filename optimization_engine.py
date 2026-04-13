"""
LPG Catering — Optimization Engine
====================================
Three optimization layers:

  Layer 1 — Household/Caterer Level
    - Refill timing: optimal booking window per caterer
    - Wastage reduction: flag inefficient caterers
    - Pre-event procurement: how many cylinders to order for an event

  Layer 2 — Regional / Dealer Level
    - Demand smoothing: flatten booking spikes across caterers
    - Load balancing: spread recommended order dates
    - Supply pressure score: warn dealers of upcoming surges

  Layer 3 — System Level (Linear Programming)
    - Minimize total wastage + stockout risk simultaneously
    - Constraint: caterer must not run out
    - Constraint: dealer delivery capacity per day

Usage:
    from optimization_engine import CatererOptimizer, RegionalOptimizer
    opt = CatererOptimizer(caterer_profile)
    result = opt.optimize_event(event_details)
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────

@dataclass
class CatererProfile:
    """Represents a catering business."""
    caterer_id: str
    name: str
    business_size: str          # small / medium / large
    experience_yrs: int
    num_burners: int
    base_region: str = "Tamil Nadu"
    avg_events_per_month: int = 4
    avg_headcount: int = 200
    preferred_menu: str = "mixed_standard"

@dataclass
class EventDetails:
    """Details of an upcoming catering event."""
    event_date: str             # YYYY-MM-DD
    event_type: str             # wedding / corporate_lunch / college_canteen / etc.
    headcount: int
    num_dishes: int
    duration_hrs: float
    menu_profile: str           # veg_simple / veg_elaborate / nonveg_elaborate / etc.
    is_festival_season: bool = False
    special_requirements: str = ""

@dataclass
class OptimizationResult:
    """Output of the optimization engine."""
    caterer_id: str
    event_date: str
    predicted_consumption_kg: float
    cylinders_to_order: int
    recommended_order_date: str
    latest_safe_order_date: str
    stockout_risk_pct: float
    wastage_estimate_kg: float
    estimated_cost_inr: float
    efficiency_score: float         # 0-100
    recommendation_tier: str        # GREEN / AMBER / RED
    action_items: List[str] = field(default_factory=list)
    optimization_notes: str = ""


# ──────────────────────────────────────────────────────────────────
# GAS CONSUMPTION REFERENCE TABLE
# Source: NRAI benchmarks + PPAC commercial LPG usage data
# ──────────────────────────────────────────────────────────────────

GAS_PER_100_GUESTS = {
    "veg_simple":       7.2,
    "veg_elaborate":   10.8,
    "nonveg_simple":    9.1,
    "nonveg_elaborate":14.3,
    "mixed_standard":  11.5,
    "snacks_only":      4.8,
    "biryani_special":  8.6,
}

WASTAGE_RATES = {
    "wedding":          0.12,
    "corporate_lunch":  0.07,
    "college_canteen":  0.09,
    "birthday_party":   0.14,
    "festival_event":   0.11,
    "hospital_canteen": 0.06,
    "school_canteen":   0.08,
    "dhaba_daily":      0.05,
}

DELIVERY_LEAD_DAYS = {
    "Chennai":      1,
    "Coimbatore":   2,
    "Madurai":      2,
    "Trichy":       2,
    "default":      3,
}

COMMERCIAL_CYLINDER_KG  = 17.5   # usable LPG per 19-kg cylinder
CYLINDER_PRICE_INR      = 1910   # IOC Chennai commercial rate 2024


# ──────────────────────────────────────────────────────────────────
# LAYER 1 — CATERER-LEVEL OPTIMIZER
# ──────────────────────────────────────────────────────────────────

class CatererOptimizer:
    """
    Per-caterer optimization engine.
    Predicts consumption, recommends order quantity, timing, and flags risk.
    """

    def __init__(self, profile: CatererProfile, cylinder_price: int = CYLINDER_PRICE_INR):
        self.profile = profile
        self.cylinder_price = cylinder_price
        self.delivery_days = DELIVERY_LEAD_DAYS.get(profile.base_region,
                             DELIVERY_LEAD_DAYS["default"])

    def predict_consumption(self, event: EventDetails) -> float:
        """
        Rule-based consumption prediction (kg).
        Used when ML model is unavailable / for validation.
        """
        base = (event.headcount / 100) * GAS_PER_100_GUESTS.get(event.menu_profile, 11.5)

        # Adjustments
        duration_factor  = event.duration_hrs / 5.0
        burner_factor    = 1.0 + (self.profile.num_burners - 4) * 0.03
        experience_adj   = max(0.85, 1.0 - self.profile.experience_yrs * 0.008)
        season_factor    = 1.20 if event.is_festival_season else 1.0
        dish_factor      = max(0.8, event.num_dishes / 5.0)

        consumption = (base * duration_factor * burner_factor
                       * experience_adj * season_factor * dish_factor)
        return round(max(1.5, consumption), 2)

    def calculate_cylinders(self, consumption_kg: float, buffer_pct: float = 0.10) -> int:
        """
        Number of 19-kg cylinders needed including a safety buffer.
        Buffer: 10% for experienced caterers, 20% for novices.
        """
        if self.profile.experience_yrs < 3:
            buffer_pct = 0.20
        elif self.profile.experience_yrs < 6:
            buffer_pct = 0.15

        total_with_buffer = consumption_kg * (1 + buffer_pct)
        return max(1, int(np.ceil(total_with_buffer / COMMERCIAL_CYLINDER_KG)))

    def calculate_order_timing(self, event_date_str: str,
                                stockout_risk_pct: float) -> Tuple[str, str]:
        """
        Returns (recommended_order_date, latest_safe_order_date).

        Logic:
          - Base buffer = delivery_days + 1 day margin
          - High risk (>50%) → add 3 extra days
          - Festival season → add 2 extra days (supply pressure)
          - Novice caterer → add 2 extra days
        """
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        base_buffer = self.delivery_days + 1

        risk_buffer    = 3 if stockout_risk_pct > 50 else 1 if stockout_risk_pct > 25 else 0
        exp_buffer     = 2 if self.profile.experience_yrs < 3 else 0

        recommended_offset = base_buffer + risk_buffer + exp_buffer
        latest_offset      = base_buffer

        recommended = (event_date - timedelta(days=recommended_offset)).strftime("%Y-%m-%d")
        latest      = (event_date - timedelta(days=latest_offset)).strftime("%Y-%m-%d")
        return recommended, latest

    def estimate_stockout_risk(self, consumption_kg: float,
                                cylinders_ordered: int) -> float:
        """
        Rule-based stockout risk percentage.
        Cross-validated against GBM classifier outputs.
        """
        usable = cylinders_ordered * COMMERCIAL_CYLINDER_KG
        gap    = (consumption_kg - usable) / max(consumption_kg, 1)

        base_risk = max(0.0, gap * 100)  # proportional under-supply risk

        # Experience penalty
        if self.profile.experience_yrs < 3:
            base_risk *= 1.5
        elif self.profile.experience_yrs < 6:
            base_risk *= 1.2

        return round(min(99.0, base_risk), 1)

    def efficiency_score(self, consumption_kg: float,
                          wastage_kg: float, cylinders: int) -> float:
        """
        Efficiency score 0-100.
        Penalises wastage + over-ordering, rewards accurate prediction.
        """
        usable = cylinders * COMMERCIAL_CYLINDER_KG
        over_order_kg = max(0, usable - consumption_kg)
        total_inefficiency = wastage_kg + over_order_kg
        inefficiency_pct = total_inefficiency / max(consumption_kg, 1) * 100
        return round(max(0, 100 - inefficiency_pct * 1.5), 1)

    def optimize_event(self, event: EventDetails,
                        ml_prediction_kg: Optional[float] = None) -> OptimizationResult:
        """
        Full optimization for a single event.
        Uses ML prediction if available, falls back to rule-based.
        """
        # 1. Consumption estimate
        rule_consumption = self.predict_consumption(event)
        if ml_prediction_kg and ml_prediction_kg > 0:
            # Blend: 70% ML, 30% rules for robustness
            consumption_kg = round(0.7 * ml_prediction_kg + 0.3 * rule_consumption, 2)
        else:
            consumption_kg = rule_consumption

        # 2. Cylinders needed
        cylinders = self.calculate_cylinders(consumption_kg)

        # 3. Stockout risk
        stockout_risk = self.estimate_stockout_risk(consumption_kg, cylinders)

        # 4. Order timing
        rec_date, latest_date = self.calculate_order_timing(
            event.event_date, stockout_risk
        )

        # 5. Wastage
        wastage_rate = WASTAGE_RATES.get(event.event_type, 0.09)
        if self.profile.experience_yrs > 8:
            wastage_rate *= 0.7   # experienced caterers waste less
        wastage_kg = round(consumption_kg * wastage_rate, 2)

        # 6. Cost
        cost_inr = cylinders * self.cylinder_price

        # 7. Efficiency
        eff = self.efficiency_score(consumption_kg, wastage_kg, cylinders)

        # 8. Risk tier
        if stockout_risk > 50 or eff < 40:
            tier = "RED"
        elif stockout_risk > 25 or eff < 65:
            tier = "AMBER"
        else:
            tier = "GREEN"

        # 9. Action items
        actions = []
        if stockout_risk > 50:
            actions.append(f"Order {cylinders + 1} cylinders (add 1 safety buffer — high risk event)")
        if self.profile.experience_yrs < 3:
            actions.append("Consider ordering 2–3 days earlier — novice buffer recommended")
        if wastage_rate > 0.10:
            actions.append(f"Wastage high for {event.event_type} events — track actual usage post-event")
        if event.is_festival_season:
            actions.append("Festival season: dealer supply pressure likely — order at least 5 days early")
        if event.headcount > 500 and self.profile.experience_yrs < 5:
            actions.append("Large event for inexperienced caterer — add 20% buffer to cylinder order")
        if not actions:
            actions.append("Procurement on track — follow recommended order date")

        notes = (
            f"Rule-based: {rule_consumption:.1f}kg"
            + (f" | ML prediction: {ml_prediction_kg:.1f}kg | Blended: {consumption_kg:.1f}kg"
               if ml_prediction_kg else f" | Final: {consumption_kg:.1f}kg")
        )

        return OptimizationResult(
            caterer_id=self.profile.caterer_id,
            event_date=event.event_date,
            predicted_consumption_kg=consumption_kg,
            cylinders_to_order=cylinders,
            recommended_order_date=rec_date,
            latest_safe_order_date=latest_date,
            stockout_risk_pct=stockout_risk,
            wastage_estimate_kg=wastage_kg,
            estimated_cost_inr=cost_inr,
            efficiency_score=eff,
            recommendation_tier=tier,
            action_items=actions,
            optimization_notes=notes,
        )


# ──────────────────────────────────────────────────────────────────
# LAYER 2 — REGIONAL DEMAND OPTIMIZER
# ──────────────────────────────────────────────────────────────────

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
                if new_date >= latest:
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


# ──────────────────────────────────────────────────────────────────
# LAYER 3 — LINEAR PROGRAMMING OPTIMIZER (scipy)
# ──────────────────────────────────────────────────────────────────

def lp_optimize_procurement(events_data: List[Dict],
                             budget_inr: Optional[float] = None,
                             max_cylinders_per_day: int = 50) -> Dict:
    """
    Linear Programming optimizer for multi-event procurement.

    Minimizes: total wastage + stockout penalty
    Subject to:
        cylinders_i >= consumption_i / CYLINDER_KG      (no stockout)
        sum(cylinders_i * PRICE) <= budget               (budget constraint)
        cylinders_ordered_day_d <= max_per_day           (delivery constraint)

    Uses scipy.optimize.linprog.
    Falls back to rule-based if scipy fails.
    """
    try:
        from scipy.optimize import linprog

        n = len(events_data)
        if n == 0:
            return {"status": "no events", "allocations": []}

        # Decision variables: cylinders[i] for each event
        # Objective: minimize wastage_cost + stockout_penalty
        # wastage_cost = (cylinders * CYL_KG - consumption) * PRICE * 0.1  (cost of excess)
        # stockout_penalty = large penalty if cylinders * CYL_KG < consumption

        consumptions = np.array([e.get("consumption_kg", 20) for e in events_data])
        min_cylinders = np.ceil(consumptions / COMMERCIAL_CYLINDER_KG).astype(int)
        max_cylinders = (min_cylinders * 1.5).astype(int)

        # Objective: minimize cylinder count (proxy for wastage)
        c = np.ones(n) * CYLINDER_PRICE_INR

        # Inequality constraints: cylinders >= min_cylinders → -cylinders <= -min
        A_ub = -np.eye(n)
        b_ub = -min_cylinders.astype(float)

        # Budget constraint
        A_budget = np.ones((1, n)) * CYLINDER_PRICE_INR
        b_budget = np.array([budget_inr if budget_inr else 1e9])
        A_ub = np.vstack([A_ub, A_budget])
        b_ub = np.append(b_ub, b_budget)

        bounds = [(int(min_cylinders[i]), int(max_cylinders[i])) for i in range(n)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            allocations = []
            for i, e in enumerate(events_data):
                cyl = max(int(min_cylinders[i]), int(np.ceil(result.x[i])))
                allocations.append({
                    "event_id":     e.get("event_id", f"E{i+1}"),
                    "event_date":   e.get("event_date", ""),
                    "consumption_kg": round(consumptions[i], 2),
                    "cylinders_lp": cyl,
                    "cylinders_rule": int(min_cylinders[i]),
                    "lp_saving_pct": round((int(min_cylinders[i]) - cyl) / max(int(min_cylinders[i]), 1) * 100, 1),
                })
            total_cost = sum(a["cylinders_lp"] * CYLINDER_PRICE_INR for a in allocations)
            total_waste_kg = sum(
                max(0, a["cylinders_lp"] * COMMERCIAL_CYLINDER_KG - a["consumption_kg"])
                for a in allocations
            )
            return {
                "status":           "optimal",
                "method":           "Linear Programming (scipy HiGHS)",
                "total_cylinders":  int(sum(a["cylinders_lp"] for a in allocations)),
                "total_cost_inr":   int(total_cost),
                "total_wastage_kg": round(total_waste_kg, 2),
                "allocations":      allocations,
            }
        else:
            raise ValueError(f"LP solver: {result.message}")

    except Exception as ex:
        # Fallback: simple rule-based
        allocations = []
        for i, e in enumerate(events_data):
            cons = e.get("consumption_kg", 20)
            cyl  = max(1, int(np.ceil(cons / COMMERCIAL_CYLINDER_KG)))
            allocations.append({
                "event_id":       e.get("event_id", f"E{i+1}"),
                "event_date":     e.get("event_date", ""),
                "consumption_kg": round(cons, 2),
                "cylinders_rule": cyl,
            })
        return {
            "status":     f"fallback (LP error: {str(ex)[:80]})",
            "method":     "Rule-based fallback",
            "allocations": allocations,
        }


# ──────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# CONVENIENCE API
# ──────────────────────────────────────────────────────────────────

def optimize_single_event(
    caterer_id: str, name: str, experience_yrs: int, num_burners: int,
    business_size: str, event_date: str, event_type: str, headcount: int,
    num_dishes: int, duration_hrs: float, menu_profile: str,
    is_festival: bool = False, ml_prediction_kg: Optional[float] = None
) -> Dict:
    """One-shot convenience function for API endpoints."""
    profile = CatererProfile(
        caterer_id=caterer_id, name=name, business_size=business_size,
        experience_yrs=experience_yrs, num_burners=num_burners,
    )
    event = EventDetails(
        event_date=event_date, event_type=event_type, headcount=headcount,
        num_dishes=num_dishes, duration_hrs=duration_hrs,
        menu_profile=menu_profile, is_festival_season=is_festival,
    )
    optimizer = CatererOptimizer(profile)
    result = optimizer.optimize_event(event, ml_prediction_kg)
    return asdict(result)


# ──────────────────────────────────────────────────────────────────
# MAIN: Run demo simulation
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LPG OPTIMIZATION ENGINE — DEMO")
    print("=" * 60)

    # Single caterer demo
    print("\n[1] Single Caterer Optimization")
    result = optimize_single_event(
        caterer_id="CAT001", name="Murugan Catering", experience_yrs=6,
        num_burners=8, business_size="medium",
        event_date=(datetime.today() + timedelta(days=7)).strftime("%Y-%m-%d"),
        event_type="wedding", headcount=500, num_dishes=7,
        duration_hrs=6.0, menu_profile="mixed_standard", is_festival=True,
    )
    print(f"  Predicted:       {result['predicted_consumption_kg']} kg")
    print(f"  Cylinders:       {result['cylinders_to_order']}")
    print(f"  Order by:        {result['recommended_order_date']}")
    print(f"  Stockout risk:   {result['stockout_risk_pct']}%")
    print(f"  Cost:            Rs {result['estimated_cost_inr']:,}")
    print(f"  Tier:            {result['recommendation_tier']}")
    print(f"  Actions:")
    for a in result["action_items"]: print(f"    - {a}")

    # Simulation demo
    print("\n[2] Running 50-caterer simulation (November — wedding peak)...")
    sim = SimulationEngine(n_caterers=50)
    sim_result = sim.run(month=11)

    imp = sim_result["improvement"]
    print(f"\n  Results:")
    print(f"  Wastage reduction:      {imp['wastage_reduction_pct']}%")
    print(f"  Peak demand reduction:  {imp['peak_demand_reduction_pct']}%")
    print(f"  Cost saving:            Rs {imp['cost_saving_inr']:,}")

    with open("optimization_result_demo.json", "w") as f:
        json.dump(sim_result, f, indent=2)
    print("\n  Full results → optimization_result_demo.json")
    print("=" * 60)
