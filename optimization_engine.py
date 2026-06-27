"""
LPG Catering — Optimization Engine
====================================
Three optimization layers exposed from the lpg_catering package.
"""

from lpg_catering.optimization.caterer import (
    CatererProfile,
    EventDetails,
    OptimizationResult,
    CatererOptimizer,
    optimize_single_event
)
from lpg_catering.optimization.regional import RegionalOptimizer
from lpg_catering.optimization.lp_solver import lp_optimize_procurement
from lpg_catering.optimization.simulation import SimulationEngine
from lpg_catering.config import CYLINDER_PRICE_INR

if __name__ == "__main__":
    import json
    from datetime import datetime, timedelta
    print("=" * 60)
    print("LPG OPTIMIZATION ENGINE — DEMO (Wrapper)")
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
    print("\n  Full results -> optimization_result_demo.json")
    print("=" * 60)
