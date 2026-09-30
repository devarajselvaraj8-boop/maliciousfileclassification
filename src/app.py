"""
FastAPI application for Static PE Malware Classification (MLCS-HACK-16).
Provides a simple, elegant Web Dashboard and REST endpoints for static PE file analysis.
SECURITY: Strictly static feature analysis - NO file execution.
"""
import os
import sys
import time
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    load_model,
    predict_from_vector,
    load_preprocessor,
    validate_pe_bytes,
    predict_from_pe_bytes,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS
)
import ember

app = FastAPI(
    title="Malicious File Classification API & Dashboard (MLCS-HACK-16)",
    description="Static Windows PE Malware Classification using EMBER 2018 v2 features.",
    version="1.3.0"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")

_loaded_models = {}
_preprocessor = None
_sample_vectors = {"benign": None, "malicious": None}


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal or special chars."""
    basename = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', basename)
    return clean_name if clean_name else "sample.exe"


@app.on_event("startup")
def startup_event():
    global _loaded_models, _preprocessor, _sample_vectors
    try:
        # Load models
        for fname in ["best_model.pkl", "random_forest.pkl", "logistic_regression.pkl", "lightgbm.pkl"]:
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.isfile(fpath):
                _loaded_models[fname] = load_model(fname, models_dir=MODELS_DIR)
        
        _preprocessor = load_preprocessor(models_dir=MODELS_DIR)
        print("[*] FastAPI: Models and preprocessor loaded successfully.")

        # Cache sample vectors from X_test if available
        x_test_path = os.path.join(DATASET_DIR, "X_test.dat")
        y_test_path = os.path.join(DATASET_DIR, "y_test.dat")
        if os.path.isfile(x_test_path) and os.path.isfile(y_test_path):
            _, _, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)
            b_idx = int(list(y_test).index(0.0)) if 0.0 in y_test else 0
            m_idx = int(list(y_test).index(1.0)) if 1.0 in y_test else 0
            _sample_vectors["benign"] = [float(x) for x in X_test[b_idx]]
            _sample_vectors["malicious"] = [float(x) for x in X_test[m_idx]]
            print(f"[*] Cached sample vectors (Benign index: {b_idx}, Malicious index: {m_idx}).")
    except Exception as e:
        print(f"[!] Warning on startup: {e}")


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Vector of 2381 static EMBER PE features")
    model_name: Optional[str] = Field("best_model.pkl", description="Model file name to use")


class PredictionResponse(BaseModel):
    label: int
    classification: str
    confidence: float
    probabilities: dict
    model_used: str
    inference_time_ms: float
    security_notice: str


@app.get("/health")
def health_check():
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    model_exists = os.path.isfile(best_model_path)
    return {
        "status": "ok" if model_exists else "degraded",
        "model_loaded": model_exists,
        "feature_count": 2381,
        "zero_execution": True,
        "max_upload_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "allowed_extensions": sorted(list(ALLOWED_EXTENSIONS)),
        "available_models": [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")] if os.path.exists(MODELS_DIR) else []
    }


@app.post("/api/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("best_model.pkl")
):
    """
    Secure endpoint for REAL static PE file classification (.exe, .dll).
    """
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Please select a .exe or .dll file."}
        )

    safe_filename = sanitize_filename(file.filename)
    _, ext = os.path.splitext(safe_filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Unsupported file extension '{ext}'. Only .exe and .dll are supported."}
        )

    try:
        bytez = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to read file: {str(e)}"}
        )

    if len(bytez) == 0:
        return JSONResponse(status_code=400, content={"success": False, "error": "Uploaded file is empty (0 bytes)."})

    if len(bytez) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={"success": False, "error": f"File exceeds maximum size ({MAX_FILE_SIZE // (1024*1024)} MB)."}
        )

    is_valid_pe, pe_reason = validate_pe_bytes(bytez)
    if not is_valid_pe:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Invalid Windows PE: {pe_reason}"})

    chosen_model = model_name or "best_model.pkl"
    if chosen_model not in _loaded_models:
        chosen_model = "best_model.pkl"
    model_obj = _loaded_models.get(chosen_model)

    try:
        result = predict_from_pe_bytes(
            bytez=bytez,
            filename=safe_filename,
            model=model_obj,
            preprocessor=_preprocessor,
            model_name=chosen_model
        )
        result["model_used"] = chosen_model
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Analysis failed: {str(e)}"})


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict on pre-existing 2,381 feature vector."""
    if len(request.features) != 2381:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature dimension. Expected 2381 features, received {len(request.features)}."
        )

    t0 = time.time()
    try:
        model_name = request.model_name or "best_model.pkl"
        if model_name not in _loaded_models:
            _loaded_models[model_name] = load_model(model_name, models_dir=MODELS_DIR)
        model = _loaded_models[model_name]
        
        use_scaled = "logistic" in model_name.lower()
        result = predict_from_vector(
            vector=request.features,
            model=model,
            preprocessor=_preprocessor,
            use_scaled=use_scaled
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    elapsed_ms = (time.time() - t0) * 1000.0

    return PredictionResponse(
        label=result["label"],
        classification=result["classification"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=request.model_name or "best_model.pkl",
        inference_time_ms=round(elapsed_ms, 2),
        security_notice="Static analysis only. Zero executable execution."
    )


@app.get("/sample/{sample_type}")
def get_sample_vector(sample_type: str):
    """Retrieve pre-extracted test vector for demo purposes."""
    sample_type = sample_type.lower()
    if sample_type not in ["benign", "malicious"]:
        raise HTTPException(status_code=400, detail="sample_type must be 'benign' or 'malicious'")
    
    vec = _sample_vectors.get(sample_type)
    if vec is None:
        vec = [0.0] * 2381
    return {"type": sample_type, "features": vec, "length": len(vec)}


@app.get("/", response_class=HTMLResponse)
def simple_dashboard():
    template_path = os.path.join(PROJECT_ROOT, "templates", "index.html")
    if not os.path.exists(template_path):
        template_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend template not found. Ensure templates/index.html exists.</h3>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
