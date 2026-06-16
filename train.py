import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('C:/Users/jenifer/Downloads/crop-datasets.csv')
df.columns = df.columns.str.strip().str.lower()

crop = input("Enter Crop Name: ") 
year = int(input("Enter year of Cultivation: ")) 
season = input("Enter Season: ") 
area = float(input("Enter Area of Land: ")) 
state = input("Enter State: ") 
yield_value = float(input("Enter Yield(tonnes): ")) 

method = input("Enter the method to calculate score: ('mean' or'median'): ").strip().lower()

filtered_df = df[ 
(df['crop'].str.strip().str.lower() == crop.lower()) & 
(df['season'].str.strip().str.lower() == season.lower()) & 
(df['state'].str.strip().str.lower() == state.lower()) 
]

if len(filtered_df) > 0:
    if method == 'mean':
        baseline_yield = filtered_df[['yield', 'area']].mean()
    else:
        baseline_yield = filtered_df[['yield', 'area']].median()
else:
    if method == 'mean':
        baseline_yield = df[['yield', 'area']].mean()
    else:
        baseline_yield = df[['yield', 'area']].median()
        
if method == 'mean':
    baseline_table = df.groupby(
        ['crop', 'season', 'state']
        )[['yield', 'area']].mean()
else:
    baseline_table = df.groupby(
        ['crop', 'season', 'state']
        )[['yield', 'area']].median()

def calculate_score(row):
    try:
        baseline = baseline_table.loc[
            (row['crop'], row['season'], row['state'])
        ]
        
        baseline_yield = baseline['yield']
        baseline_area = baseline['area']
        
    except KeyError:
        if method == 'mean':
            baseline_yield = df['yield'].mean()
            baseline_area = df['area'].mean()
        else:
            baseline_yield = df['yield'].median()
            baseline_area = df['area'].median()
            
    if (
        baseline_yield == 0 or
        baseline_area == 0 or
        pd.isna(baseline_yield) or
        pd.isna(baseline_area)
    ):
        return 0
    
    yield_ratio = row['yield'] / baseline_yield
    area_ratio = row['area'] / baseline_area
    
    combined_ratio = (
        0.8 * yield_ratio +
        0.2 * area_ratio
    )
    score = combined_ratio * 50
    score = max(0, min(score, 100))
    
    return score
df['cropping_pattern_score'] = df.apply(calculate_score,axis=1)

x = df[['crop','crop_year','season','state','area','yield']]
y = df['cropping_pattern_score']
x = pd.get_dummies(x, columns=['crop','season','state'], drop_first=True)
training_cols = x.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

joblib.dump(model, "crop_pattern_model.pkl")
joblib.dump(training_cols.tolist(), "training_cols.pkl")

model = joblib.load("crop_pattern_model.pkl")
training_cols = joblib.load("training_cols.pkl")
print("Model saved successfully!")

new_data = pd.DataFrame({ 
'crop': [crop], 
'crop_year': [year], 
'season': [season], 
'area': [area], 
'state': [state], 
'yield': [yield_value]
})

new_data = pd.get_dummies(new_data) 
new_data = new_data.reindex(columns=training_cols, fill_value=0) 

predicted_score =  model.predict(new_data)[0]

print("The Crop Pattern Score:",round(predicted_score,2) ,"%") 
if predicted_score >= 70: 
    print("The crop has great cropping pattern.") 
elif predicted_score >= 60:
    print("The crop has good cropping pattern.")
elif predicted_score >= 40:
    print("The crop has average cropping pattern.")
else: 
    print("The crop has bad cropping pattern.")
