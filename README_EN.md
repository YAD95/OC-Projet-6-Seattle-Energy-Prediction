# OC — Project 6 · Predict Building Energy Consumption in Seattle

> **Random Forest · BentoML · Google Cloud Run** — Predicting the energy consumption of non-residential buildings in Seattle from their structural characteristics, without ever measuring them. Final R² = 0.74, deployed as a REST API on GCP.

---

## Table of Contents

1. [Context & Problem Statement](#context--problem-statement)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Mission 1 — ML Modeling](#mission-1--ml-modeling)
   - [Raw Dataset](#raw-dataset)
   - [Data Cleaning](#data-cleaning)
   - [Logarithmic Transformation](#logarithmic-transformation)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Data Leakage — The Avoided Trap](#data-leakage--the-avoided-trap)
   - [Feature Engineering](#feature-engineering)
   - [Final Preparation — X/y Split](#final-preparation--xy-split)
   - [Model Comparison](#model-comparison)
   - [Optimization — GridSearchCV](#optimization--gridsearchcv)
   - [Final Results & Feature Importance](#final-results--feature-importance)
5. [Mission 2 — API & Cloud Deployment](#mission-2--api--cloud-deployment)
   - [General Architecture](#general-architecture)
   - [Run the API Locally](#run-the-api-locally)
   - [Available Endpoints](#available-endpoints)
   - [Data Validation (Pydantic)](#data-validation-pydantic)
   - [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
6. [Conclusion & Limitations](#conclusion--limitations)
7. [Data](#data)

---

## Context & Problem Statement

Buildings account for approximately **40% of global energy consumption**. As part of its commitment to **carbon neutrality by 2050**, the city of Seattle requires large non-residential buildings to report their annual energy consumption through the *Benchmarking* program.

This program generates a public dataset each year containing building structural characteristics (floor area, building type, year of construction, etc.) alongside their measured consumption figures. **The problem**: for a new building, or when targeting renovation priorities, energy consumption needs to be estimated **before** any measurement takes place.

### Objective

> Train a Machine Learning model capable of **estimating the annual energy consumption (kBtu)** of any Seattle building from its structural characteristics — **without ever having measured it**. The model is then exposed via a **REST API deployed in the cloud**, accessible in production.

### Why Does This Matter?

- Identify the heaviest energy consumers **before** inspection
- Prioritize renovations within a fixed budget
- Anticipate mandatory reporting for new buildings

---

## Project Structure

```
OC-Projet-6-Anticipez-Consommation-Energie-Batiments/
│
├── Yilmaz_Daniel_1_notebooks_062026.ipynb   # Full Mission 1 notebook
├── Yilmaz_Daniel_2_presentation_062026.pptx # Methodology presentation
│
├── service.py          # BentoML API + Pydantic validation
├── save_model.py       # Model training + saving
├── bentofile.yaml      # Docker / Cloud deployment config
│
├── 2016_Building_Energy_Benchmarking.csv    # Raw Seattle 2016 dataset
├── pyproject.toml      # Project dependencies
└── uv.lock             # Exact package versions
```

---

## Tech Stack

| Domain               | Technologies                                      |
|----------------------|---------------------------------------------------|
| **ML**               | scikit-learn, Random Forest, GridSearchCV         |
| **Data**             | pandas, numpy                                     |
| **API Validation**   | Pydantic                                          |
| **Serving**          | BentoML                                           |
| **Containerization** | Docker                                            |
| **Cloud**            | Google Cloud Run, Artifact Registry               |
| **Project Management**| Poetry / uv                                      |

---

## Mission 1 — ML Modeling

### Raw Dataset

- **Source**: Seattle Building Energy Benchmarking 2016 — [data.seattle.gov](https://data.seattle.gov/resource/2bpz-gwpy.csv)
- **Initial size**: 3,376 buildings · 46 columns
- **Scope**: non-residential buildings only
- **Year of records**: 2016

The dataset contains six main feature groups:

| Category     | Variables                                                      |
|--------------|----------------------------------------------------------------|
| Structure    | Total floor area, parking, number of buildings, floors         |
| Time         | `YearBuilt` → `BuildingAge` computed                          |
| Location     | Neighborhood, ZipCode, GPS coordinates                         |
| Equipment    | `HasGas`, `HasSteam` (flags to be created)                    |
| Performance  | `ENERGYSTARScore`                                              |
| Usage        | `PrimaryPropertyType` (22 categories)                          |

---

### Data Cleaning

#### Approach: targeted and justified

Rather than applying blind methods (classic IQR → 62.9% data loss), each removal criterion was **explicitly justified** by a business logic or a physical impossibility.

**Result: 3,376 → 1,565 buildings, representing only 3.2% of effective data loss.**

```
3,376 buildings at the start
  → 1,617 after scope filtering
  → 1,565 final clean dataset
```

#### Removal criteria in detail

| Criterion removed                     | Reason                                                                      |
|---------------------------------------|-----------------------------------------------------------------------------|
| `BuildingType = Multifamily`          | Out of scope: residential buildings — ~1,700 rows                          |
| Outliers flagged by the city          | Already officially identified as erroneous — a few rows                    |
| `SiteEnergyUse = 0`                   | Likely recording error for an active building                              |
| `PropertyGFATotal = 0`                | Physically impossible: a building cannot have zero floor area              |
| `YearBuilt` outside 1900–2016         | Inconsistent years (before 1900 or after the recording date)               |
| `NumberofFloors = 99`                 | Detected in EDA: a ~20 m² building with 99 floors — physically impossible  |
| `NumberofBuildings = 0`               | Detected in EDA: impossible to have zero buildings on a plot               |

> **Note**: The last two criteria (`NumberofFloors = 99` and `NumberofBuildings = 0`) were **identified during the EDA**, not upfront. Visualizing distributions and inspecting extreme values revealed these anomalies.

#### Cleaning approach comparison

| Method                         | Data loss |
|--------------------------------|-----------|
| Targeted cleaning (retained)   | **3.2%**  |
| Blind percentile cutoffs       | 12%       |
| Classic IQR                    | 62.9%     |

---

### Logarithmic Transformation

**Target**: `SiteEnergyUse(kBtu)` → transformed to `ln(x + 1)`

#### Why transform?

The raw distribution of energy consumption is **heavily right-skewed**: the gap between a small office building and a large hospital can reach a factor of 1,000,000. This skewness causes several issues for ML algorithms:

| Before transformation              | After transformation (`ln(x+1)`)                      |
|------------------------------------|-------------------------------------------------------|
| Very right-skewed distribution     | Distribution close to a normal law                    |
| Scale from 0 to several millions   | Scale compressed to 0–~20                             |
| Algorithms blinded by extremes     | Small and large buildings contribute equally          |
| Microscopic coefficients           | More stable and interpretable relationships           |

#### Why `+1`?

`ln(0)` is mathematically undefined. Adding 1 ensures that a building with 0 kBtu gives `ln(1) = 0` without a calculation error. The transformation is: **`SiteEnergyUse_log = ln(SiteEnergyUse + 1)`**.

> In production, all predictions are made in log space, then **back-transformed** using `exp(y_pred) - 1` to return a value in kBtu/year that is interpretable by the end user.

---

### Exploratory Data Analysis (EDA)

Three key visualizations guided the modeling decisions:

#### 1. Target distribution (QUANT)
- Log-normal distribution after transformation
- Range: 0 to 20, no visible extreme outlier
- Confirms the logarithmic transformation is appropriate

#### 2. Floor area vs. consumption (QUANT × QUANT)
- **Strong monotonic increasing relationship** between `PropertyGFATotal` and `SiteEnergyUse_log`
- The larger the building, the higher its consumption
- Foreshadows the dominant importance of this variable in feature importance

#### 3. Consumption by building type (QUAL × QUANT)
- Warehouses and hospitals display significantly higher medians than offices
- High inter-category heterogeneity → justifies including `PrimaryPropertyType`

> **Key insight**: Floor area (`GFATotal`) and building type together explain most of the variance — a conclusion confirmed by feature importance at the end of the project (71% for GFATotal alone).

---

### Data Leakage — The Avoided Trap

#### What is Data Leakage?

Data leakage means including in the training features columns that **directly derive from the target variable**. The model then learns to "cheat" — achieving an R² close to 1.0 on training data, but becoming completely useless in production where those columns do not exist for a new building.

> **Analogy**: giving the answers to a student during an exam. They score 100/100, but learned nothing.

#### Solution applied

**Columns removed** (they derive from or describe `SiteEnergyUse`):

| Removed column             | Removed column             |
|----------------------------|----------------------------|
| `Electricity(kWh)`         | `Electricity(kBtu)`        |
| `NaturalGas(therms)`       | `NaturalGas(kBtu)`         |
| `SteamUse(kBtu)`           | `SiteEUI(kBtu/sf)`         |
| `SourceEUI(kBtu/sf)`       | `TotalGHGEmissions`        |
| `GHGEmissionsIntensity`    |                            |

**Features created BEFORE removal** (structural info, no leakage):

| Feature    | Logic                                                                            |
|------------|----------------------------------------------------------------------------------|
| `HasGas`   | `NaturalGas(kBtu) > 0 → 1` — captures WHETHER the building uses gas, not how much |
| `HasSteam` | `SteamUse(kBtu) > 0 → 1` — same logic for steam                                 |

> **Golden rule**: Never include in X any information that will NOT be available at actual prediction time.

---

### Feature Engineering

Four transformations were applied to enrich the existing data:

#### A. `BuildingAge` — Time
```
Formula : BuildingAge = 2016 − YearBuilt
Example : YearBuilt = 1990 → BuildingAge = 26
```
An absolute date has no direct meaning for an ML algorithm. Age is a continuous, interpretable quantity: an older building is generally less well insulated.

#### B. `IsMultiUse` — Usage structure
```
Formula : 1 if SecondUseType ≠ "None", else 0
Example : SecondUseType = "Hotel" → IsMultiUse = 1
```
A mixed-use building (offices + retail, hotel + restaurant, etc.) consumes energy differently from a single-use building. This binary flag captures that information without introducing leakage.

#### C. `HasGas` & `HasSteam` — Equipment
```
Formula : NaturalGas(kBtu) > 0 → HasGas = 1
Example : NaturalGas = 500 → HasGas = 1
```
The goal is to know **whether** the building is connected to gas, not **how much** it consumes. The quantity would be leakage; the binary flag is structural information.

#### D. Redundancy removal — Spearman correlation
```
Criterion : |r| ≥ 0.85 → column removed
Example   : PropertyGFABuilding(s) correlated at 0.978 with PropertyGFATotal → removed
```
Three columns measured the same physical reality (total floor area). Only `PropertyGFATotal`, the most complete one, is kept to avoid multicollinearity.

---

### Final Preparation — X/y Split

#### One-Hot Encoding (OHE)

The two categorical variables were encoded into binary columns:

| Variable               | Modalities   | Columns created                        |
|------------------------|--------------|----------------------------------------|
| `PrimaryPropertyType`  | 22 values    | 9 columns + infrequent (`max_categories=10`) |
| `Neighborhood`         | 14 districts | 14 columns                             |

#### Imputation

`ENERGYSTARScore` contains missing values → imputed by the **median** to retain the maximum amount of data.

#### Final result

```
X : (1,565 × 34)   →  10 numerical features + 24 OHE features
y : (1,565,)        →  SiteEnergyUse_log
Missing values : 0
```

---

### Model Comparison

**Method**: `cross_validate` with 5 folds — rigorous evaluation preventing overfitting on the test set.

| Model                    | Train R² | Test R² | Test MAE | Test RMSE | Overfit  | Verdict          |
|--------------------------|----------|---------|----------|-----------|----------|------------------|
| DummyRegressor (baseline)| —        | -0.001  | 1.019    | 1.280     | No       | Useless          |
| LinearRegression (Ridge) | 0.596    | 0.559   | 0.645    | 0.849     | Slight   | Too simple       |
| GradientBoosting         | 0.749    | 0.624   | 0.586    | 0.784     | Moderate | —                |
| **Random Forest**        | **0.959**| **0.697**| **0.502**| **0.703** | High    | **Selected**     |

**Why Random Forest?**

The only model capable of capturing **non-linear relationships** between features. A test R² of 0.697 means the model explains ~70% of the consumption variance across buildings. The train/test gap (0.959 vs 0.697) indicates strong overfitting → addressed by GridSearchCV.

---

### Optimization — GridSearchCV

`GridSearchCV` automatically tests all hyperparameter combinations with cross-validation (5 folds each time), then returns the best one.

#### Two-step strategy

**Step 1 — Small test grid** (~10 combinations)

```python
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [5, 10],
    "min_samples_split": [2, 5]
}
# → 8 combinations × 5 folds = 40 fits
# Best R²: 0.7024
# Finding: max_depth=10 >> max_depth=5
```

**Step 2 — Full optimization grid** (~500 combinations)

```python
param_grid = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10, 15, 20]
}
# → 100 combinations × 5 folds = 500 fits
# Best R²: 0.7043
```

**Optimal hyperparameters:**

```python
n_estimators      = 500
max_depth         = 15
min_samples_split = 10
```

---

### Final Results & Feature Importance

#### Final metrics

| Metric        | Baseline | Optimized | Direction   |
|---------------|----------|-----------|-------------|
| **R² test**   | 0.697    | **0.74**  | ↑ toward 1.0|
| **MAE**       | 0.502    | **0.498** | ↓ toward 0  |
| **RMSE**      | 0.703    | **0.695** | ↓ toward 0  |
| R² train      | 0.959    | 0.899     | Overfitting reduced |

> A MAE of ~0.50 on the log target corresponds to an **error of roughly ×1.6 on the actual kBtu consumption** — perfectly acceptable for preventive estimation.

#### Feature Importance

| Feature                         | Importance  |
|---------------------------------|-------------|
| `PropertyGFATotal`              | **71.16%**  |
| `ENERGYSTARScore`               | 8.37%       |
| `PrimaryPropertyType_Warehouse` | 3.66%       |
| `BuildingAge`                   | 3.64%       |
| `HasGas`                        | 1.71%       |
| `NumberofFloors`                | 1.50%       |
| Others (28 features)            | 9.9%        |

**Breakdown:**

- **Dominant (~79%)**: `PropertyGFATotal` + `ENERGYSTARScore` — these 2 features explain most of the model
- **Useful (~11%)**: `Warehouse`, `BuildingAge`, `HasGas`, `NbFloors` — modest but relevant contributions
- **Noise (~10%)**: all `Neighborhood_*` columns + rare types — importance < 0.01 each

---

## Mission 2 — API & Cloud Deployment

### General Architecture

```
save_model.py
    ↓
BentoML Store (model saved locally)
    ↓
bentoml build → Bento package
    ↓
bentoml containerize → Docker Image
    ↓
GCP Artifact Registry (image storage)
    ↓
GCP Cloud Run (publicly accessible API)
```

### Run the API Locally

```bash
# 1. Clone the repo
git clone https://github.com/YAD95/OC-Projet-6-Anticipez-Consommation-Energie-Batiments.git
cd OC-Projet-6-Anticipez-Consommation-Energie-Batiments

# 2. Install dependencies
pip install bentoml scikit-learn numpy pandas pydantic

# 3. Train and save the model
python save_model.py

# 4. Start the API locally
bentoml serve service:SeattleEnergyService

# 5. Test via Swagger
# → http://localhost:3000/docs
```

### Available Endpoints

| Endpoint   | Method | Description                              |
|------------|--------|------------------------------------------|
| `/health`  | GET    | API status + loaded model info           |
| `/predict` | POST   | Consumption prediction (kBtu/year)       |

### Example `/predict` request

```json
{
  "NumberofBuildings": 1,
  "NumberofFloors": 5,
  "PropertyGFATotal": 15000,
  "PropertyGFAParking": 2000,
  "ENERGYSTARScore": 72,
  "DefaultData": 0,
  "BuildingAge": 30,
  "IsMultiUse": 0,
  "HasGas": 1,
  "HasSteam": 0,
  "BuildingType": "NonResidential",
  "PrimaryPropertyType": "Office",
  "Neighborhood": "DOWNTOWN"
}
```

### Expected response

```json
{
  "prediction_log": 15.2341,
  "prediction_kBtu": 4123456.78,
  "unit": "kBtu/year",
  "model": "seattle_energy_model:latest"
}
```

> `prediction_log` is the prediction in log space. `prediction_kBtu` is the back-transformed value (`exp(prediction_log) - 1`) — the figure that is actually usable by the end user.

### Data Validation (Pydantic)

The API automatically rejects inconsistent input before even calling the model:

| Field               | Constraint                                                                             |
|---------------------|----------------------------------------------------------------------------------------|
| `NumberofBuildings` | > 0                                                                                    |
| `NumberofFloors`    | > 0                                                                                    |
| `ENERGYSTARScore`   | Between 0 and 100                                                                      |
| `BuildingAge`       | Between 0 and 200 years                                                                |
| `BuildingType`      | One of: `NonResidential`, `Residential`, `Nonresidential COS`, `SPS-District K-12`    |

### GCP Cloud Run Deployment

```bash
# 1. Build the Bento
bentoml build

# 2. Containerize (force linux/amd64 for Cloud Run)
bentoml containerize seattle_energy_service:latest --opt platform=linux/amd64

# 3. Authenticate Docker with GCP
gcloud auth configure-docker europe-west1-docker.pkg.dev

# 4. Tag and push the image
docker tag seattle_energy_service:latest \
  europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest

docker push \
  europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest

# 5. Deploy to Cloud Run
gcloud run deploy seattle-api \
  --image europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 3000
```

---

## Conclusion & Limitations

### Final model

```
Algorithm : RandomForestRegressor (GridSearchCV optimized)
R² test   : 0.74   →  the model explains 74% of consumption variance
MAE       : 0.50   →  error of ~×1.6 on actual kBtu consumption
RMSE      : 0.70
```

### What the model does well

- Captures non-linear relationships between features
- Robust to extreme feature values
- Clearly identifies the heaviest energy consumers
- Clean methodology: zero data leakage, rigorous cross-validation
- Interpretable via feature importance

### Identified limitations

- **Overfitting**: R² train 0.90 vs test 0.74 — persistent despite GridSearch
- **Over-reliance on one feature**: 71% of importance rests on `PropertyGFATotal` alone
- **Missing data**: local weather, insulation level, number of occupants
- **Ceiling at 0.74**: not resolved by hyperparameter tuning — the limiting factor is feature richness
- **Neighborhood columns** contribute almost nothing to the model (<0.01% each)

### Future improvements

- Test **GradientBoosting / XGBoost** to reduce overfitting
- Enrich features with: weather data, renovation year, insulation level, occupant count
- Advanced feature engineering: floor area / floors ratio, occupancy density
- Reduce overfitting via regularization or tree pruning

### Full project pipeline

```
EDA → Log Transform → Targeted Cleaning → Data Leakage → Feature Engineering
  → X/y Split → Model Comparison → GridSearchCV → Feature Importance
    → BentoML API → Cloud Run Deployment
```

---

## Data

- **Source**: [Seattle Building Energy Benchmarking — Data.Seattle.gov](https://data.seattle.gov/resource/2bpz-gwpy.csv)
- **Included file**: `2016_Building_Energy_Benchmarking.csv` — included in the repo to allow full project reproduction
- **Scope**: non-residential buildings in Seattle, 2016 data

---

*Project completed as part of the **Data Scientist** program — OpenClassrooms · Daniel YILMAZ · June 2026*
