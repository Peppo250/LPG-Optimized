import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from lpg_catering.config import (
    GAS_PER_100_GUESTS,
    WASTAGE_RATES,
    DELIVERY_LEAD_DAYS,
    COMMERCIAL_CYLINDER_KG,
    CYLINDER_PRICE_INR
)

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
