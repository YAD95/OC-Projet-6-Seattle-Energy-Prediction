import pandas as pd
import numpy as np
import bentoml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# ── 1. CHARGEMENT ──────────────────────────────────────────────
df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# ── 2. NETTOYAGE ───────────────────────────────────────────────
df = df[~df["BuildingType"].str.contains("Multifamily")]
df = df[df["Outlier"].isna()]
df = df[df["SiteEnergyUse(kBtu)"] > 0]
df = df[df["PropertyGFATotal"] > 0]
df = df[df["NumberofFloors"] > 0]
df = df[(df["YearBuilt"] >= 1900) & (df["YearBuilt"] <= 2016)]

df["SiteEnergyUselog"] = np.log(df["SiteEnergyUse(kBtu)"] + 1)
df.drop(columns=["Outlier", "Comments", "TaxParcelIdentificationNumber"], inplace=True)

df["SecondLargestPropertyUseType"] = df["SecondLargestPropertyUseType"].fillna("Aucun")
df["ThirdLargestPropertyUseType"]  = df["ThirdLargestPropertyUseType"].fillna("Aucun")

# ── 3. FEATURE ENGINEERING ─────────────────────────────────────
df["BuildingAge"] = 2016 - df["YearBuilt"]
df["IsMultiUse"]  = ((df["SecondLargestPropertyUseType"] != "Aucun") |
                     (df["ThirdLargestPropertyUseType"]  != "Aucun")).astype(int)
df["HasGas"]   = (df["NaturalGas(kBtu)"]  > 0).astype(int)
df["HasSteam"] = (df["SteamUse(kBtu)"]    > 0).astype(int)

# ── 4. SUPPRESSION LEAKAGE + COLONNES INUTILES ─────────────────
leakage_cols = [
    "Electricity(kWh)", "Electricity(kBtu)", "NaturalGas(therms)",
    "NaturalGas(kBtu)", "SteamUse(kBtu)", "SiteEnergyUse(kBtu)",
    "SiteEnergyUseWN(kBtu)", "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
    "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)",
    "TotalGHGEmissions", "GHGEmissionsIntensity"
]
df.drop(columns=leakage_cols, errors="ignore", inplace=True)

cols_texte_inutiles = [
    "PropertyName", "Address", "City", "State",
    "ListOfAllPropertyUseTypes", "LargestPropertyUseType",
    "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType",
    "ComplianceStatus", "YearsENERGYSTARCertified", "YearBuilt"
]
cols_admin = ["OSEBuildingID", "DataYear",
              "SecondLargestPropertyUseTypeGFA", "ThirdLargestPropertyUseTypeGFA"]
df.drop(columns=cols_texte_inutiles + cols_admin, errors="ignore", inplace=True)

df["DefaultData"] = df["DefaultData"].astype(int)
df["Neighborhood"] = df["Neighborhood"].str.upper().str.strip()
df = df[df["NumberofFloors"] != 99]
df = df[df["NumberofBuildings"] > 0]

# ── 5. TARGET + FEATURES ───────────────────────────────────────
y = df["SiteEnergyUselog"]
X = df.drop(columns=["SiteEnergyUselog"])

# ── 6. TRAIN/TEST SPLIT (random_state=42 comme dans le notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 7. ONE-HOT ENCODING ────────────────────────────────────────
cols_a_encoder = ["BuildingType", "PrimaryPropertyType", "Neighborhood"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_train = ohe.fit_transform(X_train[cols_a_encoder])
encoded_test  = ohe.transform(X_test[cols_a_encoder])
encoded_cols  = ohe.get_feature_names_out(cols_a_encoder)

X_train = X_train.drop(columns=cols_a_encoder).reset_index(drop=True)
X_test  = X_test.drop(columns=cols_a_encoder).reset_index(drop=True)
X_train = pd.concat([X_train, pd.DataFrame(encoded_train, columns=encoded_cols)], axis=1)
X_test  = pd.concat([X_test,  pd.DataFrame(encoded_test,  columns=encoded_cols)], axis=1)

# ── 8. IMPUTATION MÉDIANE ──────────────────────────────────────
for col in X_train.columns:
    if X_train[col].isnull().any():
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

# ── 9. MEILLEUR MODÈLE (best_params_ GridSearch du notebook) ───
# max_depth=15, n_estimators=500, min_samples_split=10
best_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,  # ← vrais params !
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# ── 10. VÉRIFICATION RAPIDE ────────────────────────────────────
from sklearn.metrics import r2_score
r2_test = r2_score(y_test, best_model.predict(X_test))
print(f"✅ R² test : {r2_test:.4f} (attendu ~0.70)")

# ── 11. SAUVEGARDE BENTOML ─────────────────────────────────────
saved = bentoml.sklearn.save_model(
    "seattle_energy_model",
    best_model,
    custom_objects={
        "ohe": ohe,
        "feature_names": list(X_train.columns)
    },
    metadata={
        "r2_test": round(r2_test, 4),
        "best_params": {"n_estimators": 500, "max_depth": 15, "min_samples_split": 10},
        "target": "SiteEnergyUselog",
        "description": "Projet Seattle - Prédiction conso énergétique bâtiments non-résidentiels"
    }
)
print(f"✅ Modèle sauvegardé : {saved}")