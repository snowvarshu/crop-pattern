from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

dataset = pd.read_csv("crop-datasets.csv")
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
    (dataset['crop'].str.strip().str.lower() == data.crop.lower().strip()) &
    (dataset['season'].str.strip().str.lower() == data.season.lower().strip()) &
    (dataset['state'].str.strip().str.lower() == data.state.lower().strip())
]
    if matching_records.empty:
        return {
        "error": "No matching crop-season-state combination found in dataset.",
        "crop": data.crop,
        "season": data.season,
        "state": data.state
    }
    
    baseline_yield = round(
    matching_records['yield'].mean(), 2
)

    record_count = len(matching_records)
    new_data = pd.DataFrame({
        "crop": [data.crop.lower().strip()],
        "crop_year": [data.crop_year],
        "season": [data.season.lower().strip()],
        "state": [data.state.lower().strip()],
        "area": [data.area],
        "yield": [data.yield_val],
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
    "dataset_match": True,
    "method_used": selected_method
    }

    
@app.get("/")
def home():
    return {
        "message": "Crop Pattern Prediction API Running Successfully!"
    }
    
    app = WsgiToAsgi(flask_app)
    