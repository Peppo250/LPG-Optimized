# LPG Catering Intelligence System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end Machine Learning and Operations Research system designed to predict event-level LPG cylinder consumption, optimize caterer procurement, and forecast regional demand peaks for commercial caterers and LPG dealers in India.

---

## 📖 Table of Contents
1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Screenshots](#-screenshots)
5. [Tech Stack](#-tech-stack)
6. [Installation](#-installation)
7. [Quick Start](#-quick-start)
8. [Usage & Code Examples](#-usage--code-examples)
9. [Project Structure](#-project-structure)
10. [FAQ](#-faq)
11. [Roadmap](#-roadmap)
12. [Acknowledgements](#-acknowledgements)
13. [License](#-license)

---

## 🎯 Project Overview & Motivation

Large-scale food caterers in India operate in a high-pressure, low-margin environment. Running out of gas mid-event is catastrophic (cooking stops, guest reputation is damaged), while over-ordering locks up critical working capital in unused commercial cylinders. Furthermore, LPG dealers experience massive supply shocks when hundreds of caterers place last-minute orders during peak wedding seasons (e.g., Diwali in November or Pongal in January).

This system provides a **three-layer optimization framework** that solves this problem:
1. **Predict:** Blends Gradient Boosting (GBM) model predictions with rule-based physics formulas to predict consumption.
2. **Optimize:** Customizes safety buffers based on the caterer's experience, flags stockout risks, and calculates the exact order dates.
3. **Forecast:** Aggregates individual orders into regional demand curves and automatically shifts low-risk bookings 1–2 days earlier to flatten dealer delivery spikes.

---

## ✨ Key Features

* 🧠 **Blended ML Predictions:** Combines a Gradient Boosting Regressor ($R^2=0.9972$) with physics floor calculations to prevent out-of-distribution model errors.
* 📦 **Operations Research Solvers:** Utilizes SciPy's HiGHS linear programming solver to allocate cylinders under strict budget constraints.
* 📈 **Load Balancing (Smoothing):** Automatically shifts schedules of low-risk bookings to flatten regional peak demand spikes by up to 28%.
* 📊 **Interactive Operations Dashboard:** Modern dark-theme SPA frontend displaying real-time metrics, predictive inputs, model performance charts, and regional forecasts.
* 🛠️ **Unified Developer CLI:** Single runner orchestrator script (`run_project.py`) to manage data engineering, training, testing, and API server hosting.
* 🔒 **Leakage-Free Classifiers:** Stockout classifier trained on a reduced 40-feature set (experience metrics removed) to guarantee realistic ROC-AUC ($0.7865$) performance.

---

## 🏗️ System Architecture

The following diagram illustrates the flow of data from ingestion through the three optimization layers:

```
[ Raw Gov't Price Data ] \
[ Kaggle Event Wastage ]  +--> [ data_pipeline.py ] --> Normalised Event Data [0, 1]
[ Archana's Recipes    ] /
                                  |
                                  v
                         [ train_final.py ] ---> Fitted joblib Model Binaries
                                  |
                                  v
+---------------------------------+---------------------------------+
|                        [ API SERVER (api.py) ]                    |
|                                 |                                 |
|   v   Layer 1: Caterer-Level    v   Layer 2: Regional Smoothing   v   Layer 3: Scipy LP   
|  [CatererOptimizer]             |  [RegionalOptimizer]            |  [lp_optimize_procurement]
|  - Predicts Event Consumption   |  - Aggregates 30-day demand     |  - Minimizes costs & waste
|  - Blends ML + Physics floor    |  - Heuristically shifts bookings|  - Subject to budgets and  
|  - Experience safety buffer     |    early to avoid dealer peaks  |    capacity constraints
+---------------------------------+---------------------------------+
                                  |
                                  v
                       [ Interactive Dashboard ]
```

---

## 📷 Screenshots

### 1. Main Analytics Dashboard
Displays dataset summaries, predictive feature importances, and model metrics:
![Main Analytics Dashboard](docs/assets/Screenshot%202026-06-27%20173538.png)

### 2. Single Event Predictor
Takes caterer profiles and event details to recommend cylinder quantities and order timelines:
![Single Event Predictor Form](docs/assets/Screenshot%202026-06-27%20173606.png)

### 3. Multi-Event LP Refill Queue
Runs multi-event linear programming to allocate cylinder supply under fixed budgets:
![Multi-Event LP Optimizer Queue](docs/assets/Screenshot%202026-06-27%20173623.png)

### 4. Regional Dealer Forecast Simulation
Generates 30-day regional simulations comparing raw booking spikes against smoothed, shifted demand:
![Regional Forecast Simulation](docs/assets/Screenshot%202026-06-27%20173719.png)

---

## 💻 Tech Stack

* **Language:** Python 3.11+
* **Data Processing & ML:** `pandas`, `numpy`, `scikit-learn`, `joblib`
* **Mathematical Optimization:** `scipy.optimize` (HiGHS Simplex Solver)
* **Visualizations:** `matplotlib` (training summaries), `Chart.js` (dashboard charting)
* **Web APIs:** `fastapi`, `uvicorn`, `pydantic`
* **Frontend:** Vanilla HTML5, Vanilla CSS3 (custom variables visual system), Javascript (ES6)
* **Testing:** `unittest`, `httpx` (API endpoints mocking)
* **Containerization:** Docker, Docker Compose

---

## ⚙️ Installation

### Local Virtual Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/Peppo250/LPG-Optimized.git
   cd LPG-Optimized
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Deployment
Build and start the FastAPI service inside a container:
```bash
docker-compose up --build
```
*The API container binds to port `8000` with hot-reloading enabled.*

---

## 🚀 Quick Start

Get the system up and running locally in three commands:

```bash
# 1. Compile datasets, train ML models, and execute the unit test suite
python run_project.py --all

# 2. Spin up the FastAPI server
python run_project.py --api

# 3. Open dashboard.html in your default web browser to view the UI
# File URL format: file:///C:/path/to/LPG-Optimized/dashboard.html
```

---

## 💡 Usage & Code Examples

### 1. Single Caterer Optimization API
Calculate consumption and ordering parameters for an event:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "caterer_id": "CAT001",
    "caterer_name": "Murugan Catering",
    "experience_yrs": 8,
    "num_burners": 8,
    "business_size": "medium",
    "event_date": "2026-11-15",
    "event_type": "wedding",
    "headcount": 500,
    "num_dishes": 7,
    "duration_hrs": 6.0,
    "menu_profile": "mixed_standard",
    "is_festival_season": true
  }'
```

### 2. Custom Optimization Invocation (Python)
Import and run the Layer 1 buffer optimizer inside Python:
```python
from lpg_catering.optimization.caterer import CatererProfile, EventDetails, CatererOptimizer

# Define profiles
profile = CatererProfile("CAT001", "Murugan Grand", "medium", experience_yrs=6, num_burners=8)
event = EventDetails("2026-11-15", "wedding", headcount=400, num_dishes=6, duration_hrs=5.0, menu_profile="mixed_standard")

# Run optimization
optimizer = CatererOptimizer(profile)
result = optimizer.optimize_event(event, ml_prediction_kg=85.0)

print(f"Cylinders to Order: {result.cylinders_to_order}")
print(f"Recommended Order Date: {result.recommended_order_date}")
```

---

## 📂 Project Structure

```
LPG-Optimized/
├── .github/workflows/          ← CI/CD Actions (runs tests on pushes/PRs)
├── css/
│   └── style.css               ← Visual layout system for the dashboard
├── data/
│   ├── raw/                    ← Place 3 raw Gov't/Kaggle CSV files here
│   ├── processed/              ← Cleaned intermediate data
│   └── final/                  ← Normalised final datasets and scalers
├── docs/
│   ├── assets/                 ← Application screenshots
│   └── ARCHITECTURE.md         ← Technical design and LP formulations
├── js/
│   └── app.js                  ← SPA charting and API connection script
├── lpg_catering/               ← Internal Python package
│   ├── config.py               ← Centralized mappings and constants
│   └── optimization/           
│       ├── caterer.py          ← Layer 1 (Caterer buffer optimizer)
│       ├── regional.py         ← Layer 2 (Dealer smoothing and curve)
│       ├── lp_solver.py        ← Layer 3 (SciPy Linear Programming)
│       └── simulation.py       ← Simulation engine
├── tests/
│   ├── test_optimization.py    ← Optimization algorithms unit tests
│   └── test_api.py             ← Integration tests for FastAPI endpoints
├── api.py                      ← Entry point for FastAPI REST service
├── dashboard.html              ← Thin SPA html dashboard (loads CSS/JS)
├── data_pipeline.py            ← Entry point for data preprocessing
├── train_final.py              ← Entry point for model training
├── run_project.py              ← Orchestrator (pipeline, train, test, api)
├── Dockerfile                  ← Container setup configuration
├── docker-compose.yml          ← Compose service configuration
└── requirements.txt            ← Pinned dependency packages
```

---

## ❓ FAQ

#### Why did you choose a 70/30 ML-Rule blend?
Pure machine learning models are powerful but can output physically impossible predictions under extreme out-of-distribution inputs. A 30% blend of rule-based thermodynamics acts as a "physics floor", guaranteeing the caterer never receives an unsafely low allocation recommendation.

#### What happens if SciPy's Linear Program solver fails?
If the LP allocator encounters an infeasible setup (such as a budget that is mathematically too small to cover the physical gas requirements), the system catches the exception and falls back to a safe, rule-based procurement allocation.

#### Why remove experience metrics from the Stockout Classifier?
The stockout target variable was originally calculated using the caterer's experience. Training a classifier with experience metrics would lead to feature leakage, where the model simply memorizes the classification rules. Removing these features lets the classifier learn genuine event signals, outputting an honest ROC-AUC score of $0.7865$.

---

## 🗺️ Roadmap

- [ ] **Scale Sensors Integration:** Build IoT APIs to connect with digital weighing scales placed under commercial cylinders for real-time remaining gas measurements.
- [ ] **Persistent Database:** Connect the FastAPI backend to a PostgreSQL database to persist caterer profiles and historical prediction outcomes.
- [ ] **Live Price Scraper:** Scraping monthly commercial LPG prices live from public Indian oil corporations (IOC/BPCL/HPCL) portals.
- [ ] **Authentication:** Implement OAuth2 security protocols to protect predictions and LP endpoint submissions.

---

## 🤝 Acknowledgements

* **Rajya Sabha Session 260 (Q&A)**: Ministry of Petroleum & Natural Gas, Government of India (Source of yearly commercial price data).
* **National Restaurant Association India (NRAI)**: Validation benchmarks for commercial kitchen LPG consumption rates.
* **Archana's Kitchen Recipe Database**: 6,800+ recipe files parsed for gas cooking intensities.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
