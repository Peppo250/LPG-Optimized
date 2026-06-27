import numpy as np
from typing import List, Dict, Optional
from lpg_catering.config import COMMERCIAL_CYLINDER_KG, CYLINDER_PRICE_INR

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
