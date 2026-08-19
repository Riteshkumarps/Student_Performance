from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pickle
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current directory:
# D:\Project\Student Performance\app

BASE_DIR = Path(__file__).resolve().parent


# Model:
# D:\Project\Student Performance\Notebook\Model.pkl

MODEL_PATH = BASE_DIR.parent / "Notebook" / "Model.pkl"


# HTML:
# D:\Project\Student Performance\app\static\templates\index.html

HTML_PATH = (
    BASE_DIR
    / "static"
    / "templates"
    / "index.html"
)


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Performance Prediction",
    description="Math Score Prediction using Machine Learning",
    version="1.0.0"
)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model.pkl not found at: {MODEL_PATH}"
    )


# ============================================================
# CHECK HTML FILE
# ============================================================

if not HTML_PATH.exists():

    raise FileNotFoundError(
        f"index.html not found at: {HTML_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)


print("==========================================")
print("Model loaded successfully!")
print(f"Model path: {MODEL_PATH}")
print(f"HTML path: {HTML_PATH}")
print("==========================================")


# ============================================================
# INPUT DATA MODEL
# ============================================================

class StudentData(BaseModel):

    gender: str

    race_ethnicity: str

    parental_level_of_education: str

    lunch: str

    test_preparation_course: str

    reading_score: float

    writing_score: float


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    return FileResponse(
        HTML_PATH
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(data: StudentData):

    # Convert received data into DataFrame

    input_data = pd.DataFrame([
        {
            "gender": data.gender,

            "race_ethnicity":
                data.race_ethnicity,

            "parental_level_of_education":
                data.parental_level_of_education,

            "lunch":
                data.lunch,

            "test_preparation_course":
                data.test_preparation_course,

            "reading_score":
                data.reading_score,

            "writing_score":
                data.writing_score
        }
    ])


    print("\nReceived student data:")
    print(input_data)


    # Make prediction

    prediction = model.predict(input_data)


    # Get predicted score

    math_score = float(prediction[0])


    # Keep score within 0-100

    math_score = max(
        0,
        min(100, math_score)
    )


    print(
        f"Predicted Math Score: {math_score}"
    )


    return {
        "success": True,
        "math_score": round(math_score, 2)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "OK",
        "model_loaded": True
    }