

The Crop Pattern Prediction System is a Machine Learning-powered web application that predicts the future cropping pattern score of agricultural land based on crop characteristics, historical yield data, cultivation season, state, and cultivated area.
The system uses a trained Random Forest Regression model to analyze historical crop production records and generate a cropping pattern score that helps assess agricultural productivity and sustainability.

The application consists of:

* Machine Learning Model (Scikit-Learn)
* FastAPI Backend
* Angular Frontend
* Google Cloud Run Deployment
* Docker Containerization
* GitHub Version Control

---

## Features

### Crop Pattern Prediction

Predicts cropping pattern score using:

* Crop Name
* Crop Year
* Season
* State
* Area
* Current Yield
* Calculation Method (Mean / Median)

### Dataset Validation

Verifies whether the given:

* Crop
* Season
* State

combination exists in the historical dataset.

### Historical Analysis

Provides:

* Number of matching records
* Historical average yield
* Dataset verification status

### Cloud Deployment

Accessible globally through Google Cloud Run.

---

## Technology Stack

### Frontend

* Angular
* TypeScript
* HTML
* CSS

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Machine Learning

* Scikit-Learn
* Random Forest Regressor
* Pandas
* NumPy
* Joblib

### Cloud & DevOps

* Docker
* Google Cloud Run
* GitHub
* Git LFS

---

# System Architecture

User Interface (Angular)

        ↓

FastAPI REST API

        ↓

Data Validation Layer

        ↓

Feature Engineering

        ↓

Random Forest Model

        ↓

Prediction Engine

       ↓

JSON Response

       ↓

Angular Dashboard

---

## Project Structure

```text
crop-pattern/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── crop_pattern_model.pkl
├── training_cols.pkl
├── crop-datasets.csv
│
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── angular.json
│   └── ...
│
└── README.md
```

---

# Dataset

## Source

Agricultural crop production dataset containing historical crop records.

### Important Attributes

| Column     | Description        |
| ---------- | ------------------ |
| Crop       | Crop Name          |
| Crop_Year  | Cultivation Year   |
| Season     | Cultivation Season |
| State      | State Name         |
| Area       | Cultivated Area    |
| Yield      | Crop Yield         |
| Production | Crop Production    |

---

## Data Preprocessing

The following preprocessing steps are performed:

### Column Cleaning

```python
df.columns = df.columns.str.strip().str.lower()
```

### Missing Value Handling

Rows with invalid data are removed.

### Categorical Encoding

Categorical features:

* Crop
* Season
* State

are transformed using:

```python
pd.get_dummies()
```

### Feature Alignment

Training columns are saved:

```python
joblib.dump(training_cols,
            "training_cols.pkl")
```

This ensures future predictions use the exact same feature layout.

---

# Machine Learning Model

## Algorithm

Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42
)
```

### Why Random Forest?

* Handles categorical data effectively
* Reduces overfitting
* Good prediction accuracy
* Robust against noisy data

---

## Training Process

### Load Dataset

```python
df = pd.read_csv("crop-datasets.csv")
```

### Feature Engineering

Convert categorical variables into numerical format.

### Split Dataset

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Train Model

```python
model.fit(X_train, y_train)
```

### Save Model

```python
joblib.dump(
    model,
    "crop_pattern_model.pkl"
)
```

---

# Model Persistence using Joblib

Joblib stores the trained model for future use without retraining.

### Save Model

```python
joblib.dump(model,
            "crop_pattern_model.pkl")
```

### Load Model

```python
model = joblib.load(
    "crop_pattern_model.pkl"
)
```

Benefits:

* Faster startup
* No retraining required
* Production-ready deployment

---

# FastAPI Backend

## Create API

```python
app = FastAPI(
    title="Crop Pattern API"
)
```

---

## Prediction Endpoint

```python
@app.post("/predict")
```

Receives:

```json
{
  "crop": "Rice",
  "crop_year": 2027,
  "season": "Kharif",
  "state": "Tamil Nadu",
  "area": 100,
  "yield": 3200,
  "method": "mean"
}
```

Returns:

```json
{
  "cropping_pattern_score": 72.54,
  "status": "Great cropping pattern",
  "historical_average_yield": 2980.12,
  "matching_records": 156,
  "method_used": "mean"
}
```

---

## API Documentation

Swagger UI:

```text
https://YOUR-CLOUD-RUN-URL/docs
```

ReDoc:

```text
https://YOUR-CLOUD-RUN-URL/redoc
```

---

# Angular Frontend

The frontend provides a user-friendly dashboard.

### Input Fields

* Crop Name
* Crop Year
* Season
* State
* Area
* Yield
* Method

### API Integration

```typescript
this.http.post(
  apiUrl,
  payload
)
```

### Output

Displays:

* Cropping Pattern Score
* Prediction Status
* Historical Yield
* Dataset Validation

---

# Docker Containerization

## Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn",
     "app:app",
     "--host",
     "0.0.0.0",
     "--port",
     "8080"]
```

---

# Google Cloud Deployment

## Build Container

```bash
docker build -t crop-pattern-api .
```

---

## Deploy to Cloud Run

```bash
gcloud run deploy crop-pattern-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated
```

---

## Deployment Architecture

Angular Frontend

↓

Google Cloud Run

↓

FastAPI Backend

↓

Random Forest Model

↓

Prediction Result

---

# GitHub Version Control

## Initialize Repository

```bash
git init
```

## Add Files

```bash
git add .
```

## Commit

```bash
git commit -m "Initial Commit"
```

## Push

```bash
git push -u origin main
```

---


# Author

Snow Varshini

Machine Learning | FastAPI | Angular | Google Cloud

---

# License

This project is intended for educational, research, and agricultural analytics purposes.
