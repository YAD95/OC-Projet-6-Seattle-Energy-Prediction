#  OC — Projet 6 · Anticipez la Consommation Énergétique des Bâtiments de Seattle

> **Random Forest · BentoML · Google Cloud Run** — Prédiction de la consommation énergétique des bâtiments non-résidentiels de Seattle à partir de leurs caractéristiques structurelles, sans jamais les avoir mesurés. R² final = 0.74, déployé en API REST sur GCP.

---

## 📋 Sommaire

1. [Contexte & Problématique](#-contexte--problématique)
2. [Structure du projet](#-structure-du-projet)
3. [Stack technique](#-stack-technique)
4. [Mission 1 — Modélisation ML](#-mission-1--modélisation-ml)
   - [Le dataset brut](#le-dataset-brut)
   - [Nettoyage des données](#nettoyage-des-données)
   - [Transformation logarithmique](#transformation-logarithmique)
   - [Analyse Exploratoire (EDA)](#analyse-exploratoire-eda)
   - [Data Leakage — Le piège évité](#data-leakage--le-piège-évité)
   - [Feature Engineering](#feature-engineering)
   - [Préparation finale — Split X/y](#préparation-finale--split-xy)
   - [Comparaison des modèles](#comparaison-des-modèles)
   - [Optimisation — GridSearchCV](#optimisation--gridsearchcv)
   - [Résultats finaux & Feature Importance](#résultats-finaux--feature-importance)
5. [Mission 2 — API & Déploiement Cloud](#-mission-2--api--déploiement-cloud)
   - [Architecture générale](#architecture-générale)
   - [Lancer l'API en local](#lancer-lapi-en-local)
   - [Endpoints disponibles](#endpoints-disponibles)
   - [Validation des données (Pydantic)](#validation-des-données-pydantic)
   - [Déploiement GCP Cloud Run](#déploiement-gcp-cloud-run)
6. [Conclusion & Limites](#-conclusion--limites)
7. [Données](#-données)

---

## 🌍 Contexte & Problématique

Les bâtiments représentent environ **40 % de la consommation d'énergie mondiale**. Dans le cadre de son engagement vers la **neutralité carbone d'ici 2050**, la ville de Seattle impose aux grands bâtiments non-résidentiels de déclarer leur consommation énergétique annuelle via le programme *Benchmarking*.

Ce programme génère chaque année un dataset public contenant les caractéristiques structurelles des bâtiments (surface, type d'usage, année de construction…) ainsi que leurs consommations mesurées. **Le problème** : pour un nouveau bâtiment, ou pour cibler des rénovations prioritaires, on a besoin d'estimer la consommation **avant** toute mesure.

### Objectif

> *Entraîner un modèle de Machine Learning capable d'**estimer la consommation énergétique annuelle (kBtu)** de tout bâtiment de Seattle à partir de ses caractéristiques structurelles — **sans jamais l'avoir mesuré**. Ce modèle est ensuite exposé via une **API REST déployée dans le cloud**, accessible en production.* 

### Pourquoi c'est utile ?

- Identifier les plus gros consommateurs **avant** inspection
- Prioriser les rénovations à budget constant
- Anticiper les déclarations obligatoires pour les nouveaux bâtiments

---

## 📁 ---Structure du projet---

```
OC-Projet-6-Anticipez-Consommation-Energie-Batiments/
│
├── Yilmaz_Daniel_1_notebooks_062026.ipynb   # Notebook complet Mission 1
├── Yilmaz_Daniel_2_presentation_062026.pptx # Présentation méthodologie
│
├── service.py          # API BentoML + validation Pydantic
├── save_model.py       # Entraînement + sauvegarde du modèle
├── bentofile.yaml      # Config déploiement Docker / Cloud
│
├── 2016_Building_Energy_Benchmarking.csv    # Dataset brut Seattle 2016
├── pyproject.toml      # Dépendances du projet
└── uv.lock             # Versions exactes des packages
```

---

##  ---Stack technique---

| Domaine              | Technologies                                      |
|----------------------|---------------------------------------------------|
| **ML**               | scikit-learn, Random Forest, GridSearchCV         |
| **Data**             | pandas, numpy                                     |
| **Validation API**   | Pydantic                                          |
| **Serving**          | BentoML                                           |
| **Containerisation** | Docker                                            |
| **Cloud**            | Google Cloud Run, Artifact Registry               |
| **Gestion de projet**| Poetry / uv                                       |

---

## 🔬 Mission 1 — Modélisation ML

### Le dataset brut

- **Source** : Seattle Building Energy Benchmarking 2016 — [data.seattle.gov](https://data.seattle.gov/resource/2bpz-gwpy.csv)
- **Taille initiale** : 3 376 bâtiments · 46 colonnes
- **Périmètre** : bâtiments non-résidentiels uniquement
- **Année des relevés** : 2016

Le dataset contient six grandes familles de features :

| Catégorie       | Variables concernées                                       |
|-----------------|------------------------------------------------------------|
| Structure       | Surface totale, parking, nombre de bâtiments, étages       |
| Temporalité     | `YearBuilt` → `BuildingAge` calculé                        |
| Localisation    | Quartier (`Neighborhood`), ZipCode, coordonnées GPS        |
| Équipements     | `HasGas`, `HasSteam` (flags à créer)                       |
| Performance     | `ENERGYSTARScore`                                          |
| Usage           | `PrimaryPropertyType` (22 catégories)                      |

---

### Nettoyage des données

#### Approche adoptée : chirurgicale et justifiée

Plutôt que d'appliquer des méthodes aveugles (IQR classique → 62.9 % de perte), chaque critère de suppression a été **explicitement justifié** par une logique métier ou une impossibilité physique.

**Résultat : 3 376 → 1 565 bâtiments, soit seulement 3.2 % de perte effective de données.**

```
3 376 bâtiments de départ
  → 1 617 après filtrage de périmètre
  → 1 565 dataset final propre
```

#### Détail des critères de suppression

| Critère supprimé                      | Raison                                                               |
|---------------------------------------|----------------------------------------------------------------------|
| `BuildingType = Multifamily`          | Hors périmètre : bâtiments résidentiels — ~1 700 lignes              |
| Outliers signalés par la mairie       | Déjà identifiés officiellement comme aberrants — quelques lignes     |
| `SiteEnergyUse = 0`                   | Erreur de relevé probable pour un bâtiment en activité              |
| `PropertyGFATotal = 0`                | Impossible physiquement : un bâtiment ne peut avoir une surface nulle |
| `YearBuilt` hors 1900–2016            | Années incohérentes (antérieures à 1900 ou postérieures au relevé)   |
| `NumberofFloors = 99`                 | Détecté en EDA : bâtiment de ~20 m² avec 99 étages → impossibe physiquement |
| `NumberofBuildings = 0`               | Détecté en EDA : impossible d'avoir zéro bâtiment sur une parcelle  |

> **Note** : Les deux derniers critères (`NumberofFloors = 99` et `NumberofBuildings = 0`) ont été **détectés lors de l'analyse exploratoire (EDA)** et non en amont. L'EDA a révélé ces anomalies grâce à la visualisation des distributions et à l'inspection des valeurs extrêmes.

#### Comparaison des approches de nettoyage

| Méthode                     | Perte de données |
|-----------------------------|-----------------|
| ✅Nettoyage ciblé (retenu) | **3.2 %**       |
| Percentiles aveugles        | 12 %            |
| IQR classique               | 62.9 %          |

---

### Transformation logarithmique

**Target** : `SiteEnergyUse(kBtu)` → transformée en `ln(x + 1)`

#### Pourquoi transformer ?

La distribution brute de la consommation énergétique est **très asymétrique à droite** : l'écart entre un petit immeuble de bureaux et un grand hôpital peut atteindre un facteur 1 000 000. Cette asymétrie pose plusieurs problèmes pour les algorithmes ML :

| Avant transformation               | Après transformation (`ln(x+1)`)                  |
|------------------------------------|---------------------------------------------------|
| Distribution très asymétrique      | Distribution proche d'une loi normale             |
| Échelle de 0 à plusieurs millions  | Échelle compressée de 0 à ~20                     |
| Algorithmes aveuglés par extrêmes  | Petits et grands bâtiments contribuent équitablement |
| Coefficients microscopiques        | Relations plus stables et interprétables          |

#### Pourquoi `+1` ?

`ln(0)` n'est pas défini mathématiquement. Ajouter 1 garantit qu'un bâtiment à 0 kBtu donne `ln(1) = 0` sans erreur de calcul. La transformation est donc : **`SiteEnergyUse_log = ln(SiteEnergyUse + 1)`**.

> ⚠️ En production, toutes les prédictions sont produites dans l'espace log, puis **ré-inversées** avec `exp(y_pred) - 1` pour restituer une valeur en kBtu/an compréhensible par l'utilisateur.

---

### Analyse Exploratoire (EDA)

Trois visualisations clés ont guidé les décisions de modélisation :

#### 1. Distribution de la target (QUANTI)
- Distribution log-normale après transformation
- Étendue : 0 à 20, sans outlier extrême visible
- Confirme que la transformation logarithmique est appropriée

#### 2. Surface vs Consommation (QUANTI × QUANTI)
- Relation **monotone croissante forte** entre `PropertyGFATotal` et `SiteEnergyUse_log`
- Plus le bâtiment est grand, plus sa consommation est élevée
- Préfigure l'importance dominante de cette variable en feature importance

#### 3. Consommation par type de bâtiment (QUALI × QUANTI)
- Les entrepôts (`Warehouse`) et les hôpitaux affichent des médianes nettement plus élevées que les bureaux
- Forte hétérogénéité inter-catégories → justifie l'inclusion du `PrimaryPropertyType`

> **Insight clé** : La surface (`GFATotal`) et le type de bâtiment expliquent à eux deux la majeure partie de la variance — conclusion confirmée par la feature importance en fin de projet (71 % pour GFATotal seul).

---

### Data Leakage — Le piège évité

#### C'est quoi le Data Leakage ?

Le data leakage consiste à inclure dans les features d'entraînement des colonnes qui **dérivent directement de la variable cible**. Le modèle apprend alors à "tricher" — il obtient un R² proche de 1.0 sur les données d'entraînement, mais est totalement inutile en production où ces colonnes n'existent pas pour un nouveau bâtiment.

> **Analogie** : donner les réponses à un étudiant pendant l'examen. Il obtient 100/100, mais n'a rien appris.

#### Notre solution appliquée

**Colonnes supprimées** (elles dérivent de `SiteEnergyUse` ou la décrivent) :

| Colonne supprimée          | Colonne supprimée          |
|----------------------------|----------------------------|
| `Electricity(kWh)`         | `Electricity(kBtu)`        |
| `NaturalGas(therms)`       | `NaturalGas(kBtu)`         |
| `SteamUse(kBtu)`           | `SiteEUI(kBtu/sf)`         |
| `SourceEUI(kBtu/sf)`       | `TotalGHGEmissions`        |
| `GHGEmissionsIntensity`    |                            |

**Features créées AVANT la suppression** (info structurelle, pas de leakage) :

| Feature      | Logique                                                                 |
|--------------|-------------------------------------------------------------------------|
| `HasGas`     | `NaturalGas(kBtu) > 0 → 1` — on capture SI le bâtiment utilise du gaz, pas combien |
| `HasSteam`   | `SteamUse(kBtu) > 0 → 1` — même logique pour la vapeur                |

> **Règle d'or** : Ne jamais inclure dans X des informations qu'on ne possédera PAS au moment de la prédiction réelle.

---

### Feature Engineering

Quatre transformations ont été appliquées pour enrichir les données existantes :

#### A. `BuildingAge` — Temporalité
```
Formule : BuildingAge = 2016 − YearBuilt
Exemple : YearBuilt = 1990 → BuildingAge = 26
```
Une date absolue n'a pas de sens direct pour un algorithme ML. Un âge est une grandeur continue interprétable : un bâtiment plus vieux est généralement moins bien isolé.

#### B. `IsMultiUse` — Structure d'usage
```
Formule : 1 si SecondUseType ≠ "None", sinon 0
Exemple : SecondUseType = "Hotel" → IsMultiUse = 1
```
Un bâtiment multi-usage (bureaux + commerces, hôtel + restaurant...) consomme différemment d'un bâtiment mono-usage. Ce flag binaire capture cette information sans créer de leakage.

#### C. `HasGas` & `HasSteam` — Équipements
```
Formule : NaturalGas(kBtu) > 0 → HasGas = 1
Exemple : NaturalGas = 500 → HasGas = 1
```
On veut savoir **si** le bâtiment est raccordé au gaz, pas **combien** il en consomme. La quantité serait du leakage, le flag binaire est une information structurelle.

#### D. Suppression des redondances — Corrélation Spearman
```
Critère : |r| ≥ 0.85 → colonne supprimée
Exemple : PropertyGFABuilding(s) corrélée à 0.978 avec PropertyGFATotal → supprimée
```
Trois colonnes mesuraient la même réalité physique (la surface totale). On conserve uniquement `PropertyGFATotal`, la plus complète, pour éviter la multicolinéarité.

---

### Préparation finale — Split X/y

#### One-Hot Encoding (OHE)

Les deux variables catégorielles ont été encodées en colonnes binaires :

| Variable               | Modalités    | Colonnes créées         |
|------------------------|--------------|-------------------------|
| `PrimaryPropertyType`  | 22 valeurs   | 9 colonnes + infrequent (`max_categories=10`) |
| `Neighborhood`         | 14 quartiers | 14 colonnes             |

#### Imputation

`ENERGYSTARScore` contient des valeurs manquantes → imputées par la **médiane** pour conserver le maximum de données.

#### Résultat final

```
X final : (1 565 × 34)    →  10 features numériques + 24 features OHE
y        : (1 565,)        →  SiteEnergyUse_log
Valeurs manquantes : 0
```

---

### Comparaison des modèles

**Méthode** : `cross_validate` avec 5 folds — évaluation rigoureuse évitant le surapprentissage sur le jeu de test.

| Modèle                  | Train R² | Test R² | Test MAE | Test RMSE | Overfit | Verdict        |
|-------------------------|----------|---------|----------|-----------|---------|----------------|
| DummyRegressor (baseline)| —       | -0.001  | 1.019    | 1.280     | Non     | ❌ Inutile     |
| LinearRegression (Ridge) | 0.596   | 0.559   | 0.645    | 0.849     | Léger   | ⚠️ Trop simple |
| GradientBoosting        | 0.749    | 0.624   | 0.586    | 0.784     | Modéré  | —              |
| **Random Forest**        | **0.959**| **0.697**| **0.502**| **0.703** | Fort   | ✅ **Retenu**  |

**Pourquoi le Random Forest ?**

Seul modèle à capturer les **relations non-linéaires** entre features. Un R² test de 0.697 signifie que le modèle explique ~70 % de la variance de consommation entre bâtiments. L'écart train/test (0.959 vs 0.697) indique un overfitting fort → traité par GridSearchCV.

---

### Optimisation — GridSearchCV

`GridSearchCV` teste automatiquement toutes les combinaisons d'hyperparamètres avec cross-validation (5 folds) à chaque fois, puis retourne la meilleure.

#### Stratégie en deux étapes

**Étape 1 — Petite grille de test** (~10 combinaisons)

```python
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [5, 10],
    "min_samples_split": [2, 5]
}
# → 8 combinaisons × 5 folds = 40 fits
# Meilleur R² : 0.7024
# Enseignement : max_depth=10 >> max_depth=5
```

**Étape 2 — Grande grille d'optimisation** (~500 combinaisons)

```python
param_grid = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10, 15, 20]
}
# → 100 combinaisons × 5 folds = 500 fits
# Meilleur R² : 0.7043
```

**Hyperparamètres optimaux :**

```python
n_estimators      = 500
max_depth         = 15
min_samples_split = 10
```

---

### Résultats finaux & Feature Importance

#### Métriques finales

| Métrique      | Baseline | Optimisé | Sens du gain     |
|---------------|----------|----------|------------------|
| **R² test**   | 0.697    | **0.74** | ↑ vers 1.0       |
| **MAE**       | 0.502    | **0.498**| ↓ vers 0         |
| **RMSE**      | 0.703    | **0.695**| ↓ vers 0         |
| R² train      | 0.959    | 0.899    | Overfitting réduit |

> Un MAE de ~0.50 sur la target log correspond à une **erreur d'un facteur ~×1.6 sur la consommation réelle en kBtu** — très correct pour de l'estimation préventive.

#### Feature Importance

| Feature                         | Importance |
|---------------------------------|-----------|
| `PropertyGFATotal`              | **71.16 %** |
| `ENERGYSTARScore`               | 8.37 %    |
| `PrimaryPropertyType_Warehouse` | 3.66 %    |
| `BuildingAge`                   | 3.64 %    |
| `HasGas`                        | 1.71 %    |
| `NumberofFloors`                | 1.50 %    |
| Autres (28 features)            | 9.9 %     |

**Analyse des résultats :**

- **Dominantes (~79 %)** : `PropertyGFATotal` + `ENERGYSTARScore` — ces 2 features expliquent l'essentiel du modèle
- **Utiles (~11 %)** : `Warehouse`, `BuildingAge`, `HasGas`, `NbFloors` — contributions modestes mais pertinentes
- **Bruit (~10 %)** : tous les `Neighborhood_*` + types rares — importance < 0.01 chacune

---

## 🚀 Mission 2 — API & Déploiement Cloud

### Architecture générale

```
save_model.py
    ↓
BentoML Store (modèle sauvegardé localement)
    ↓
bentoml build → Bento package
    ↓
bentoml containerize → Docker Image
    ↓
GCP Artifact Registry (stockage de l'image)
    ↓
GCP Cloud Run (API accessible publiquement)
```

### Lancer l'API en local

```bash
# 1. Cloner le repo
git clone https://github.com/YAD95/OC-Projet-6-Anticipez-Consommation-Energie-Batiments.git
cd OC-Projet-6-Anticipez-Consommation-Energie-Batiments

# 2. Installer les dépendances
pip install bentoml scikit-learn numpy pandas pydantic

# 3. Entraîner et sauvegarder le modèle
python save_model.py

# 4. Lancer l'API en local
bentoml serve service:SeattleEnergyService

# 5. Tester via Swagger
# → http://localhost:3000/docs
```

### Endpoints disponibles

| Endpoint   | Méthode | Description                               |
|------------|---------|-------------------------------------------|
| `/health`  | GET     | Statut de l'API + modèle chargé           |
| `/predict` | POST    | Prédiction de consommation (kBtu/an)      |

### Exemple de requête `/predict`

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

### Réponse attendue

```json
{
  "prediction_log": 15.2341,
  "prediction_kBtu": 4123456.78,
  "unit": "kBtu/an",
  "model": "seattle_energy_model:latest"
}
```

> `prediction_log` est la prédiction dans l'espace log. `prediction_kBtu` est la valeur ré-inversée (`exp(prediction_log) - 1`) — c'est le chiffre exploitable par l'utilisateur.

### Validation des données (Pydantic)

L'API rejette automatiquement les données incohérentes avant même d'appeler le modèle :

| Champ               | Contrainte                                                                            |
|---------------------|---------------------------------------------------------------------------------------|
| `NumberofBuildings` | > 0                                                                                   |
| `NumberofFloors`    | > 0                                                                                   |
| `ENERGYSTARScore`   | Entre 0 et 100                                                                        |
| `BuildingAge`       | Entre 0 et 200 ans                                                                    |
| `BuildingType`      | Parmi : `NonResidential`, `Residential`, `Nonresidential COS`, `SPS-District K-12`   |

### Déploiement GCP Cloud Run

```bash
# 1. Builder le Bento
bentoml build

# 2. Containeriser (forcer linux/amd64 pour Cloud Run)
bentoml containerize seattle_energy_service:latest --opt platform=linux/amd64

# 3. Authentifier Docker avec GCP
gcloud auth configure-docker europe-west1-docker.pkg.dev

# 4. Tagger et pousser l'image
docker tag seattle_energy_service:latest \
  europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest

docker push \
  europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest

# 5. Déployer sur Cloud Run
gcloud run deploy seattle-api \
  --image europe-west1-docker.pkg.dev/seattle-energy-497714/seattle-api/seattle-energy-predictor:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 3000
```

---

## 🏁 Conclusion & Limites

### Modèle final

```
Algorithme : RandomForestRegressor (GridSearchCV optimisé)
R² test    : 0.74   →  le modèle explique 74 % de la variance de consommation
MAE        : 0.50   →  erreur ~×1.6 sur la consommation réelle en kBtu
RMSE       : 0.70
```

### Ce que le modèle fait bien ✅

- Capture les relations non-linéaires entre features
- Robuste aux valeurs extrêmes des features
- Identifie clairement les très grands consommateurs
- Démarche propre : zéro data leakage, cross-validation rigoureuse
- Interprétable via feature importance

### Limites identifiées ⚠️

- **Overfitting** : R² train 0.90 vs test 0.74 — persistant malgré GridSearch
- **Dépendance excessive** : 71 % de l'importance repose sur une seule feature (`PropertyGFATotal`)
- **Données manquantes** : météo locale, niveau d'isolation, nombre d'occupants
- **Plafond à 0.74** : non résolu par les hyperparamètres — le facteur limitant est la richesse des features
- **Quartiers** (`Neighborhood`) n'apportent quasi rien au modèle (<0.01 % chacun)

### Pour aller plus loin 

- Tester **GradientBoosting / XGBoost** pour réduire l'overfitting
- Enrichir les features avec : données météo, année de rénovation, niveau d'isolation, nombre d'occupants
- Feature engineering avancé : ratio surface/étages, densité d'occupation
- Réduire l'overfitting via régularisation ou pruning des arbres

### Parcours complet du projet

```
EDA → Log Transform → Nettoyage ciblé → Data Leakage → Feature Engineering
  → Split X/y → Comparaison modèles → GridSearchCV → Feature Importance
    → API BentoML → Déploiement Cloud Run
```

---

## 📊 Données

- **Source** : [Seattle Building Energy Benchmarking — Data.Seattle.gov](https://data.seattle.gov/resource/2bpz-gwpy.csv)
- **Fichier inclus** : `2016_Building_Energy_Benchmarking.csv` — inclus dans le repo pour permettre une reproduction complète du projet
- **Périmètre** : bâtiments non-résidentiels de Seattle, données 2016

---

*Projet réalisé dans le cadre du parcours **Data Engineer** — OpenClassrooms · Daniel Alican YILMAZ · Juin 2026*
