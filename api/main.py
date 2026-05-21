# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 5 - Integration LLM via Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    region: str

class ExplainOutput(BaseModel):
    explication: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement du modele ---
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {list(model.classes_)}")

# --- Client Groq ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("Client Groq initialise.")

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.get("/model-info")
def model_info():
    return {
        "type": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Sexe invalide : {patient.sexe}")
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Region inconnue : {patient.region}")
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        region_enc
    ]])
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())
    confiance = "haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.4 else "faible"
    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    prompt = f"""Un modele de machine learning a analyse un patient :
- Age : {data.age} ans
- Sexe : {data.sexe}
- Region : {data.region}
- Diagnostic predit : {data.diagnostic}
- Probabilite : {int(data.probabilite * 100)}%

Explique ce diagnostic en 2-3 phrases simples en français, 
comme si tu parlais directement au patient. 
Ne dis pas que tu es une IA. Sois rassurant mais honnete."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant medical senegalais. Tu expliques les diagnostics en wolof simple et accessible. Si tu ne peux pas en wolof, utilise le français."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200
    )
    return ExplainOutput(explication=response.choices[0].message.content)

# --- Servir le frontend (Lab 6) ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")

app.mount("/static", StaticFiles(directory="frontend"), name="static")
