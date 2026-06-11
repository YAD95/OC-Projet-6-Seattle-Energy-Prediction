# service.py
import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ── Charger le modèle ─────────────────────────────────────────
model_ref = bentoml.sklearn.get("seattle_energy_model:latest")

# ── Schéma d'entrée AVEC validation ──────────────────────────
class BuildingInput(BaseModel):
    NumberofBuildings: float = Field(gt=0, description="Doit être > 0")
    NumberofFloors: float = Field(gt=0, description="Doit être > 0")
    PropertyGFATotal: float = Field(gt=0, description="Surface totale > 0")
    PropertyGFAParking: float = Field(ge=0, description="Surface parking >= 0")
    ENERGYSTARScore: float = Field(ge=0, le=100, description="Score entre 0 et 100")
    DefaultData: int = Field(ge=0, le=1, description="0 ou 1 uniquement")
    BuildingAge: float = Field(ge=0, le=200, description="Âge entre 0 et 200 ans")
    IsMultiUse: int = Field(ge=0, le=1, description="0 ou 1 uniquement")
    HasGas: int = Field(ge=0, le=1, description="0 ou 1 uniquement")
    HasSteam: int = Field(ge=0, le=1, description="0 ou 1 uniquement")
    BuildingType: str
    PrimaryPropertyType: str
    Neighborhood: str

    @field_validator("BuildingType")
    @classmethod
    def check_building_type(cls, v):
        allowed = ["NonResidential", "Residential", "Nonresidential COS", "SPS-District K-12"]
        if v not in allowed:
            raise ValueError(f"BuildingType doit être parmi : {allowed}")
        return v

    @field_validator("PropertyGFATotal")
    @classmethod
    def check_gfa_coherence(cls, v):
        if v > 5_000_000:
            raise ValueError("Surface totale irréaliste (> 5 000 000 sqft)")
        return v

# ── Le service ────────────────────────────────────────────────
@bentoml.service
class SeattleEnergyService:

    def __init__(self):
        self.model = model_ref.load_model()
        self.ohe = model_ref.custom_objects["ohe"]
        self.feature_names = model_ref.custom_objects["feature_names"]

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok", "model": "seattle_energy_model:latest"}

    @bentoml.api
    def predict(self, input: BuildingInput) -> dict:
        try:
            raw = pd.DataFrame([input.model_dump()])

            cols_a_encoder = ["BuildingType", "PrimaryPropertyType", "Neighborhood"]
            encoded = self.ohe.transform(raw[cols_a_encoder])
            encoded_cols = self.ohe.get_feature_names_out(cols_a_encoder)

            X = raw.drop(columns=cols_a_encoder).reset_index(drop=True)
            X = pd.concat([X, pd.DataFrame(encoded, columns=encoded_cols)], axis=1)

            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]

            log_pred = self.model.predict(X)
            pred_kbtu = float(np.expm1(log_pred[0]))

            return {
                "prediction_log":  round(float(log_pred[0]), 4),
                "prediction_kBtu": round(pred_kbtu, 2),
                "unit":            "kBtu/an",
                "model":           "seattle_energy_model:latest"
            }

        except Exception as e:
            return {"error": str(e)}