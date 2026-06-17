from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

dataset = pd.read_csv("C:/Users/jenifer/Downloads/crop-datasets.csv")
dataset.columns = dataset.columns.str.strip().str.lower()

app = FastAPI(title="Crop Pattern API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("crop_pattern_model.pkl")
training_cols = joblib.load("training_cols.pkl")

@app.get("/")
def home():
    return {
        "message": "Crop Pattern Prediction API Running Successfully!"
    }
    
class CropRequest(BaseModel):
    crop: str
    crop_year: int
    season: str
    state: str
    area: float
    yield_val: float = Field(..., alias="yield") 
    method : str 

@app.post("/predict")
def predict(data: CropRequest):
    selected_method = data.method.lower().strip()
    if selected_method == "mean":
        print("Using the MEAN method.")
    elif selected_method == "median":
        print("Using the MEDIAN method.")
    else:
        return {"error": "Invalid method. Please send 'mean' or 'median'."}
    
    matching_records = dataset[
    (dataset['state'].str.strip().str.lower() == data.state.lower().strip())
]
    if matching_records.empty:
        return {
        "error": "No matching state found in dataset.",
        "state": data.state
    }
    
    if selected_method == "mean":
        baseline_yield = round(
        matching_records['yield'].mean(), 2
    )
    else:
        baseline_yield = round(
        matching_records['yield'].median(), 2
    )
        
    adjusted_yield = (
    baseline_yield + data.yield_val
    ) / 2

    new_data = pd.DataFrame({
        "crop": [data.crop.lower().strip()],
        "crop_year": [data.crop_year],
        "season": [data.season.lower().strip()],
        "state": [data.state.lower().strip()],
        "area": [data.area],
        "yield": [adjusted_yield],
        "method": [data.method.lower().strip()]
    })
    

    new_data = pd.get_dummies(new_data)

    new_data = new_data.reindex(
        columns=training_cols,
        fill_value=0
    )

    score = float(model.predict(new_data)[0])
    
    if score >= 70:
        status = "The crop has great cropping pattern"
    elif score >= 60:
        status = "The crop has good cropping pattern"
    elif score >= 40:
        status = "The crop has average cropping pattern"
    else:
        status = "Bad cropping pattern"
        
    return {
    "cropping_pattern_score": round(score, 2),
    "status": status
    }

