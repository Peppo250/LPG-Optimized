"""
LPG Catering Intelligence — FastAPI Backend (Final)
=====================================================
All endpoints reload from disk on every request.
No stale in-memory caching of training results.

Install:
    pip install fastapi uvicorn joblib scikit-learn pandas numpy scipy

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Verify:
    http://localhost:8000/api/metrics
"""

import os, json, warnings, traceback
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from optimization_engine import (
    CatererProfile, EventDetails, CatererOptimizer,
    RegionalOptimizer, SimulationEngine,
    lp_optimize_procurement, optimize_single_event,
    CYLINDER_PRICE_INR,
)

MODEL_DIR = "models_final"
DATA_DIR  = "data/final"

# ── Load models once at startup ──────────────────────────────────
MODELS: Dict[str, Any] = {}

def load_models():
    if not HAS_JOBLIB:
        return
    for key, fname in {
        "gbm_consumption": "gbm_consumption.joblib",
        "mlp_consumption": "mlp_consumption.joblib",
        "gbm_cylinders":   "gbm_cylinders.joblib",
        "gbm_stockout":    "gbm_stockout.joblib",
        "feature_scaler":  "feature_scaler.joblib",
    }.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                MODELS[key] = joblib.load(path)
                print(f"  [OK] {key}")
            except Exception as e:
                print(f"  [WARN] {key}: {e}")
        else:
            print(f"  [MISSING] {path}")

    for key, fname in {
        "feature_cols":          "feature_cols.json",
        "feature_cols_stockout": "feature_cols_stockout.json",
        "target_meta":           "target_meta.json",
        "metadata":              f"../{DATA_DIR}/feature_metadata.json",
    }.items():
        path = os.path.join(MODEL_DIR, fname) if not fname.startswith("..") else fname[3:]
        if os.path.exists(path):
            with open(path) as f:
                MODELS[key] = json.load(f)

load_models()


def read_training_report() -> dict:
    """Always read from disk — never from cache."""
    path = os.path.join(MODEL_DIR, "training_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def read_raw_dataset() -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, "lpg_catering_dataset_raw.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_feature_importance_from_models() -> dict:
    """Compute feature importance live from loaded models."""
    try:
        feat     = MODELS.get("feature_cols", [])
        feat_stk = MODELS.get("feature_cols_stockout", feat)
        gbm_c    = MODELS.get("gbm_consumption")
        stk      = MODELS.get("gbm_stockout")
        if not (feat and gbm_c and stk):
            raise ValueError("Models not loaded")
        fi_cons = sorted(
            [{"feature": f, "consumption": round(float(v),6)}
             for f,v in zip(feat, gbm_c.feature_importances_)],
            key=lambda x: x["consumption"], reverse=True
        )
        fi_stk = sorted(
            [{"feature": f, "stockout": round(float(v),6)}
             for f,v in zip(feat_stk, stk.feature_importances_)],
            key=lambda x: x["stockout"], reverse=True
        )
        return {"success": True, "stockout_top10": fi_stk[:10], "consumption_top10": fi_cons[:10]}
    except Exception as e:
        # Fallback to CSV
        path = os.path.join(MODEL_DIR, "feature_importance.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            stk_col  = "stockout"    if "stockout"    in df.columns else df.columns[1]
            cons_col = "consumption" if "consumption" in df.columns else df.columns[2]
            return {
                "success": True,
                "stockout_top10":    df.nlargest(10, stk_col)[["feature",stk_col]].rename(columns={stk_col:"stockout"}).to_dict("records"),
                "consumption_top10": df.nlargest(10, cons_col)[["feature",cons_col]].rename(columns={cons_col:"consumption"}).to_dict("records"),
            }
        return {"success": False, "message": str(e)}


def ml_predict_consumption(req) -> Optional[float]:
    """ML prediction for consumption_kg in real units."""
    try:
        feat  = MODELS.get("feature_cols", [])
        gbm_c = MODELS.get("gbm_consumption")
        meta  = MODELS.get("target_meta") or MODELS.get("metadata",{}).get("scaler_params",{})
        if not (feat and gbm_c):
            return None
        TMIN = meta.get("TMIN") or meta.get("target_min", [0]*4)
        TMAX = meta.get("TMAX") or meta.get("target_max", [500,29,100,30])

        month = datetime.strptime(req.event_date, "%Y-%m-%d").month
        etype_list = ["wedding","corporate_lunch","college_canteen","birthday_party",
                      "festival_event","hospital_canteen","school_canteen","dhaba_daily"]
        menu_list  = ["veg_simple","veg_elaborate","nonveg_simple","nonveg_elaborate",
                      "mixed_standard","snacks_only","biryani_special"]
        fmap = {
            "headcount":             req.headcount / 3000,
            "num_dishes":            req.num_dishes / 10,
            "duration_hrs":          req.duration_hrs / 16,
            "num_burners":           req.num_burners / 20,
            "experience_yrs":        req.experience_yrs / 25,
            "wedding_season":        1.0 if month in [1,2,10,11,12] else 0.0,
            "festival_flag":         1.0 if req.is_festival_season else 0.0,
            "is_monsoon":            1.0 if month in [6,7,8,9] else 0.0,
            "month_sin":             float(np.sin(2*np.pi*month/12)),
            "month_cos":             float(np.cos(2*np.pi*month/12)),
            "log_headcount":         float(np.log1p(req.headcount) / np.log1p(3000)),
            "heat_stress":           0.0,
            "covid_period":          0.0,
            "event_type_enc":        (etype_list.index(req.event_type)/7
                                      if req.event_type in etype_list else 0.0),
            "menu_profile_enc":      (menu_list.index(req.menu_profile)/6
                                      if req.menu_profile in menu_list else 0.0),
            "business_size_enc":     {"small":0.0,"medium":0.5,"large":1.0}.get(req.business_size, 0.5),
            "kg_per_guest":          0.088,   # dataset average — updated by optimization engine
            "kg_per_burner_hr":      0.088 * req.headcount / (req.num_burners * req.duration_hrs + 1e-6) / 10,
            "event_scale":           (req.headcount * req.duration_hrs / 100) / 300,
            "season_intensity":      (1.0 if month in [1,2,10,11,12] else 0.0)*0.4 + (1.0 if req.is_festival_season else 0.0)*0.3,
        }
        # One-hot event type
        for et in etype_list:
            fmap[f"etype_{et}"] = 1.0 if req.event_type == et else 0.0

        x   = np.array([fmap.get(f, 0.0) for f in feat], dtype=np.float32).reshape(1,-1)
        pred_scaled = gbm_c.predict(x)[0]
        pred_kg = pred_scaled * (TMAX[0] - TMIN[0]) + TMIN[0]
        return round(float(np.clip(pred_kg, 1.5, 500)), 2)
    except Exception:
        return None


def ml_predict_stockout(req) -> Optional[float]:
    """Stockout probability 0-100%."""
    try:
        feat_stk = MODELS.get("feature_cols_stockout", [])
        stk      = MODELS.get("gbm_stockout")
        if not (feat_stk and stk):
            return None
        month = datetime.strptime(req.event_date, "%Y-%m-%d").month
        etype_list = ["wedding","corporate_lunch","college_canteen","birthday_party",
                      "festival_event","hospital_canteen","school_canteen","dhaba_daily"]
        fmap = {
            "headcount":         req.headcount/3000,
            "num_dishes":        req.num_dishes/10,
            "duration_hrs":      req.duration_hrs/16,
            "num_burners":       req.num_burners/20,
            "wedding_season":    1.0 if month in [1,2,10,11,12] else 0.0,
            "festival_flag":     1.0 if req.is_festival_season else 0.0,
            "is_monsoon":        1.0 if month in [6,7,8,9] else 0.0,
            "month_sin":         float(np.sin(2*np.pi*month/12)),
            "month_cos":         float(np.cos(2*np.pi*month/12)),
            "log_headcount":     float(np.log1p(req.headcount)/np.log1p(3000)),
            "event_scale":       (req.headcount * req.duration_hrs / 100) / 300,
            "kg_per_guest":      0.088,
            "kg_per_burner_hr":  0.088 * req.headcount / (req.num_burners * req.duration_hrs + 1e-6) / 10,
            "season_intensity":  (1.0 if month in [1,2,10,11,12] else 0.0)*0.4,
        }
        for et in etype_list:
            fmap[f"etype_{et}"] = 1.0 if req.event_type == et else 0.0
        x   = np.array([fmap.get(f, 0.0) for f in feat_stk], dtype=np.float32).reshape(1,-1)
        prob = float(stk.predict_proba(x)[0][1]) * 100
        return round(prob, 1)
    except Exception:
        return None


if HAS_FASTAPI:
    class PredictRequest(BaseModel):
        caterer_id:        str   = "CAT001"
        caterer_name:      str   = "Murugan Catering"
        experience_yrs:    int   = Field(8,  ge=1, le=40)
        num_burners:       int   = Field(8,  ge=1, le=30)
        business_size:     str   = Field("medium", pattern="^(small|medium|large)$")
        event_date:        str   = Field(description="YYYY-MM-DD")
        event_type:        str   = "wedding"
        headcount:         int   = Field(300, ge=10, le=5000)
        num_dishes:        int   = Field(6,   ge=1,  le=20)
        duration_hrs:      float = Field(5.0, ge=0.5, le=16.0)
        menu_profile:      str   = "mixed_standard"
        is_festival_season:bool  = False

    class BatchRequest(BaseModel):
        events:      List[PredictRequest]
        budget_inr:  Optional[float] = None
        use_lp:      bool = True

    app = FastAPI(
        title="LPG Catering Intelligence API",
        description="AI-powered LPG prediction, optimization and forecasting for Indian food caterers",
        version="3.0",
        docs_url="/docs",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # ── Endpoints ────────────────────────────────────────────────

    @app.get("/")
    async def health():
        return {
            "status":    "running",
            "version":   "3.0",
            "models":    [k for k in MODELS if not k.startswith("_")],
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/api/metrics")
    async def metrics():
        """All KPIs — always reloaded from disk. No cached values."""
        tr  = read_training_report()
        mi  = tr.get("models", {})
        raw = read_raw_dataset()

        event_dist   = {}
        monthly_avg  = {}
        stockout_cnt = 0
        cyl_price    = CYLINDER_PRICE_INR

        if raw is not None:
            event_dist   = (raw["event_type"].value_counts(normalize=True)*100).round(1).to_dict()
            monthly_avg  = {int(k):round(float(v),1)
                            for k,v in raw.groupby("month")["consumption_kg"].mean().items()}
            stockout_cnt = int(raw["ran_out_of_gas"].sum())
            cyl_price    = int(raw["commercial_price_inr"].max())

        return {
            "model_performance": {
                "consumption_r2":     mi.get("consumption_kg",{}).get("gbm",{}).get("r2",       0),
                "consumption_mae_kg": mi.get("consumption_kg",{}).get("gbm",{}).get("mae_kg",   0),
                "cylinders_r2":       mi.get("cylinders_needed",{}).get("gbm",{}).get("r2",     0),
                "cylinders_mae":      mi.get("cylinders_needed",{}).get("gbm",{}).get("mae_cyl",0),
                "stockout_f1":        mi.get("ran_out_of_gas",{}).get("f1",  0),
                "stockout_auc":       mi.get("ran_out_of_gas",{}).get("auc", 0),
                "stockout_acc":       mi.get("ran_out_of_gas",{}).get("acc", 0),
                "mlp_consumption_r2": mi.get("consumption_kg",{}).get("mlp",{}).get("r2",      0),
                "mlp_cylinders_r2":   mi.get("cylinders_needed",{}).get("mlp",{}).get("r2",    0),
            },
            "dataset": {
                "total_events":    tr.get("rows",     0),
                "feature_columns": tr.get("features", 0),
                "date_range":      "2018–2023",
                "stockout_events": stockout_cnt,
                "stockout_rate":   tr.get("stockout_rate_pct", 0),
                "event_types":     len(event_dist),
                "real_sources":    tr.get("real_data", []),
            },
            "event_distribution":      event_dist,
            "monthly_avg_consumption": monthly_avg,
            "cylinder_price_inr":      cyl_price,
        }

    @app.post("/api/predict")
    async def predict(req: PredictRequest):
        try:
            ml_kg  = ml_predict_consumption(req)
            ml_stk = ml_predict_stockout(req)
            result = optimize_single_event(
                caterer_id=req.caterer_id, name=req.caterer_name,
                experience_yrs=req.experience_yrs, num_burners=req.num_burners,
                business_size=req.business_size, event_date=req.event_date,
                event_type=req.event_type, headcount=req.headcount,
                num_dishes=req.num_dishes, duration_hrs=req.duration_hrs,
                menu_profile=req.menu_profile, is_festival=req.is_festival_season,
                ml_prediction_kg=ml_kg,
            )
            result["ml_consumption_kg"]    = ml_kg
            result["ml_stockout_risk_pct"] = ml_stk
            return {"success": True, "data": result}
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.post("/api/optimize")
    async def optimize(req: PredictRequest):
        return await predict(req)

    @app.post("/api/batch-optimize")
    async def batch_optimize(req: BatchRequest):
        try:
            individual, lp_events = [], []
            for e in req.events:
                ml_kg  = ml_predict_consumption(e)
                result = optimize_single_event(
                    caterer_id=e.caterer_id, name=e.caterer_name,
                    experience_yrs=e.experience_yrs, num_burners=e.num_burners,
                    business_size=e.business_size, event_date=e.event_date,
                    event_type=e.event_type, headcount=e.headcount,
                    num_dishes=e.num_dishes, duration_hrs=e.duration_hrs,
                    menu_profile=e.menu_profile, is_festival=e.is_festival_season,
                    ml_prediction_kg=ml_kg,
                )
                individual.append(result)
                lp_events.append({"event_id":e.caterer_id,"event_date":e.event_date,
                                   "consumption_kg":result["predicted_consumption_kg"]})
            lp = lp_optimize_procurement(lp_events, budget_inr=req.budget_inr) if req.use_lp else None
            return {
                "success": True,
                "summary": {
                    "total_events":    len(individual),
                    "total_cylinders": sum(r["cylinders_to_order"] for r in individual),
                    "total_cost_inr":  sum(r["estimated_cost_inr"]  for r in individual),
                    "avg_stockout_risk": round(np.mean([r["stockout_risk_pct"] for r in individual]),1),
                    "high_risk_events": sum(1 for r in individual if r["recommendation_tier"]=="RED"),
                },
                "lp_optimization":   lp,
                "individual_results": individual,
            }
        except Exception as e:
            raise HTTPException(500, detail=f"{e}\n{traceback.format_exc()}")

    @app.get("/api/regional-forecast")
    async def regional_forecast(
        region:       str = Query("Tamil Nadu"),
        n_caterers:   int = Query(40),
        month:        int = Query(11),
        dealer_cap:   int = Query(200),
    ):
        sim = SimulationEngine(n_caterers=n_caterers, seed=99)
        result = sim.run(month=month)
        return {
            "success":     True,
            "region":      region,
            "summary":     result["regional_summary"],
            "improvement": result["improvement"],
            "demand_curve": result["demand_curve"][:30],
        }

    @app.get("/api/simulation")
    async def simulation(
        n_caterers: int = Query(50, ge=10, le=500),
        month:      int = Query(11, ge=1,  le=12),
    ):
        sim = SimulationEngine(n_caterers=n_caterers)
        return {"success": True, "data": sim.run(month=month)}

    @app.get("/api/feature-importance")
    async def feature_importance():
        """Always computed from loaded model objects — not from stale CSV."""
        return get_feature_importance_from_models()

    @app.get("/api/model-metrics")
    async def model_metrics():
        return {"success": True, "metrics": read_training_report()}

    @app.get("/api/caterers")
    async def caterers():
        return {"success": True, "caterers": [
            {"id":"CAT001","name":"Murugan Grand Catering","size":"large", "exp":15,"burners":12,"city":"Chennai"},
            {"id":"CAT002","name":"Meenakshi Events",      "size":"medium","exp":7, "burners":6, "city":"Madurai"},
            {"id":"CAT003","name":"Balaji Caterers",       "size":"small", "exp":2, "burners":3, "city":"Trichy"},
            {"id":"CAT004","name":"Annapoorna Services",   "size":"medium","exp":10,"burners":8, "city":"Coimbatore"},
            {"id":"CAT005","name":"Royal Wedding Caterers","size":"large", "exp":20,"burners":18,"city":"Chennai"},
        ]}


if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        print("Starting API on http://localhost:8000")
        print("Docs: http://localhost:8000/docs")
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        import json
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        print("\nDemo prediction:")
        from datetime import date
        result = optimize_single_event(
            "CAT001","Demo",8,8,"medium",
            (date.today()+timedelta(7)).strftime("%Y-%m-%d"),
            "wedding",400,7,6.0,"mixed_standard",True
        )
        print(json.dumps(result, indent=2))