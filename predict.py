import pandas as pd
import joblib

df = pd.read_csv("C:/Users/jenifer/Downloads/crop-datasets.csv")
df.columns = df.columns.str.strip().str.lower()

model = joblib.load(
    "crop_pattern_model.pkl"
)

training_cols = joblib.load(
    "training_cols.pkl"
)

crop = input("Enter Crop Name: ") 
year = int(input("Enter year of Cultivation: ")) 
season = input("Enter Season: ") 
area = float(input("Enter Area of Land: ")) 
state = input("Enter State: ") 
yield_value = float(input("Enter Yield(tonnes): ")) 

method = input("Enter the method to calculate score: ('mean' or'median'): ").strip().lower()

matching_records = df[
    (df['state'].str.lower() == state.lower())
]

if matching_records.empty:
    print("\nError: This state does not exist in the dataset.")
    exit()

if method == "mean":
    baseline_yield = matching_records['yield'].mean()
    baseline_area = matching_records['area'].mean()

else:
    baseline_yield = matching_records['yield'].median()
    baseline_area = matching_records['area'].median()


if baseline_yield == 0 or baseline_area == 0:
    print("Baseline yield or area is zero.")
    exit()

yield_ratio = yield_value / baseline_yield
area_ratio = area / baseline_area

score_factor = (
    0.8 * yield_ratio +
    0.2 * area_ratio
)


adjusted_yield = baseline_yield * score_factor
adjusted_area = baseline_area * score_factor


adjusted_yield = (
    baseline_yield +
    yield_value
) / 2

new_data = pd.DataFrame({ 
'crop': [crop], 
'crop_year': [year], 
'season': [season], 
'area': [adjusted_area], 
'state': [state], 
'yield': [adjusted_yield]
})

new_data = pd.get_dummies(new_data) 
new_data = new_data.reindex(columns=training_cols, fill_value=0) 

predicted_score = model.predict(new_data)[0]

print("The Crop Pattern Score:",round(predicted_score,2) ,"%") 
if predicted_score >= 70: 
    print("The crop has great cropping pattern.") 
elif predicted_score >= 60:
    print("The crop has good cropping pattern.")
elif predicted_score >= 40:
    print("The crop has average cropping pattern.")
else: 
    print("The crop has bad cropping pattern.")
