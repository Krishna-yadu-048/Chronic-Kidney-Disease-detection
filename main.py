"""
main.py — FastAPI app. Run: uvicorn main:app --reload
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.preprocessing import preprocess_single
from src.model import load_model

app = FastAPI(title="CKD Predictor")
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")
model = load_model("outputs/model.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict_ckd(
    request: Request,
    SerumCreatinine:         str = Form(default=""),
    GFR:                     str = Form(default=""),
    MuscleCramps:            str = Form(default=""),
    FastingBloodSugar:       str = Form(default=""),
    ProteinInUrine:          str = Form(default=""),
    Itching:                 str = Form(default=""),
    SerumElectrolytesSodium: str = Form(default=""),
    HemoglobinLevels:        str = Form(default=""),
    NauseaVomiting:          str = Form(default=""),
    PhysicalActivity:        str = Form(default=""),
):
    input_data = {
        "SerumCreatinine":         SerumCreatinine,
        "GFR":                     GFR,
        "MuscleCramps":            MuscleCramps,
        "FastingBloodSugar":       FastingBloodSugar,
        "ProteinInUrine":          ProteinInUrine,
        "Itching":                 Itching,
        "SerumElectrolytesSodium": SerumElectrolytesSodium,
        "HemoglobinLevels":        HemoglobinLevels,
        "NauseaVomiting":          NauseaVomiting,
        "PhysicalActivity":        PhysicalActivity,
    }

    X          = preprocess_single(input_data, save_dir="outputs")
    prediction = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]

    result = {
        "label":      "CKD Detected"    if prediction == 1 else "No CKD Detected",
        "is_ckd":     bool(prediction == 1),
        "confidence": round(float(max(proba)) * 100, 1),
        "ckd_prob":   round(float(proba[1]) * 100, 1),
        "no_ckd_prob":round(float(proba[0]) * 100, 1),
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
