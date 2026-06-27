# System Architecture & Technical Design

This document details the underlying math, heuristics, and software engineering architecture implemented in the **LPG Catering Intelligence System**.

---

## 1. Data Ingestion & Preprocessing

The system relies on three real-world data sources (government reports, Kaggle event records, recipe databases) combined with commercial kitchen rules.

```
+------------------------------+
| Gov't Commercial Price Data  | ----+
+------------------------------+     |
                                     |
+------------------------------+     +---> [data_pipeline.py]
|  Kaggle Food Wastage Dataset  | ----+          |
+------------------------------+     |          |
                                     |          v
+------------------------------+     |   [Feature Engineering]
|  6,800+ Recipes (Archana's)  | ----+          |
+------------------------------+                v
                                         Normalised Dataset
                                         - Scaled [0, 1]
                                         - median imputed
```

### Feature Engineering
The raw data is transformed into model-ready features through several engineering steps:
* **`log_headcount`**: Natural log of the guest headcount to reduce the strong right-skewness of event scales:
  $$\text{log\_headcount} = \ln(\text{headcount} + 1)$$
* **`month_sin` / `month_cos`**: Trigonometric encoding of event calendar months to represent cyclic temporal behaviors:
  $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
* **`season_intensity`**: A linear combination mapping the wedding season, festival seasonality, and historical demand multiplier:
  $$\text{season\_intensity} = 0.4 \cdot \text{wedding\_season} + 0.3 \cdot \text{festival\_flag} + 0.3 \cdot (\text{demand\_multiplier} - 1.0)$$
* **`dish_load`**: Number of dishes multiplied by recipe-derived gas intensity:
  $$\text{dish\_load} = \text{num\_dishes} \cdot \text{gas\_intensity\_per\_serving}$$

---

## 2. Machine Learning Architecture

The system trains three specialized models:
1. **Consumption Regressor (`consumption_kg`)**: Blends Gradient Boosting (winner) and MLP Neural Network architectures to output predicted gas consumption in kilograms.
2. **Cylinder Needs Regressor (`cylinders_needed`)**: Predicts the discrete count of standard 19-kg cylinders needed for an event.
3. **Stockout Classifier (`ran_out_of_gas`)**: Predicts the probability (0–100%) that the caterer will run out of gas mid-event.

### Preventing Feature Leakage
The stockout target variable (`ran_out_of_gas`) is derived using variables like `experience_yrs`. To ensure the classifier learns from genuine operational signals rather than memorizing the derivation rules, `experience_yrs` and `novice_peak_season` are dropped from the training set. This reduces the feature count from 42 to 40, resulting in a robust, honest ROC-AUC score of **0.7865**.

---

## 3. Three-Layer Optimization System

The core value of the system lies in the sequential application of three optimization layers:

```
+-------------------------------------------------------+
|  Layer 1: Caterer-Level Buffer Heuristic              |
|  - Blends ML prediction (70%) with physical rules     |
|  - Buffers cylinder counts by experience (10% - 20%)  |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  Layer 2: Regional Demand Smoothing Heuristic         |
|  - Aggregates daily deliveries into 30-day curves     |
|  - Shifts low-risk orders early to avoid spikes       |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  Layer 3: Scipy Linear Programming                    |
|  - Minimizes cylinders under budget constraints       |
+-------------------------------------------------------+
```

### Layer 1: Experience-Adjusted Safety Buffers
Pure ML models can produce physically implausible values under out-of-distribution conditions. The system applies a blended consumption estimate:
$$\text{predicted\_consumption} = 0.7 \cdot \text{ml\_predict} + 0.3 \cdot \text{rule\_predict}$$

Safety cylinder buffers are added based on the caterer's experience tier:
$$\text{buffer\_pct} = 
\begin{cases} 
20\% & \text{experience} < 3 \text{ years} \\
15\% & 3 \le \text{experience} < 6 \text{ years} \\
10\% & \text{experience} \ge 6 \text{ years}
\end{cases}$$

$$\text{cylinders\_to\_order} = \left\lceil \frac{\text{predicted\_consumption} \cdot (1 + \text{buffer\_pct})}{\text{usable\_kg\_per\_cylinder}} \right\rceil$$

### Layer 2: Heuristic Demand Smoothing
Dealers experience spikes when multiple caterers book on the same delivery date. The regional smoothing algorithm:
1. Detects daily delivery peaks that exceed 80% of regional dealer daily capacity.
2. Identifies orders on those peak days with low stockout risks ($<15\%$).
3. Shifts those orders 1–2 days earlier, ensuring they do not breach the latest safe ordering date (delivery days + safety margins).

### Layer 3: Linear Programming Solver
When procuring cylinders across multiple events under a fixed budget, the system models the optimization as a linear programming task:

**Objective Function:**
Minimize total procurement cost (which serves as a proxy for minimizing gas wastage):
$$\min_{x} \sum_{i=1}^{n} c_i \cdot x_i$$
Where:
* $x_i$ is the number of cylinders allocated to event $i$.
* $c_i$ is the commercial price per cylinder.

**Constraints:**
1. **No Stockout Constraint:** The usable cylinder capacity allocated must exceed predicted consumption:
   $$x_i \ge \frac{\text{consumption}_i}{17.5} \quad \forall i \in \{1, \dots, n\}$$
2. **Budget Constraint:** Total expenditure must not exceed the caterer's budget limit:
   $$\sum_{i=1}^{n} c_i \cdot x_i \le \text{budget}$$
3. **Solver Bounds:** The cylinders allocated must reside within safety bounds:
   $$\text{min\_cylinders}_i \le x_i \le 1.5 \cdot \text{min\_cylinders}_i$$

The solver uses the Scipy HiGHS simplex method. If the linear program is infeasible (e.g. the budget is too low), the system automatically defaults to a robust rule-based safety allocation.
