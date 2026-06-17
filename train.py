import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("C:/Users/jenifer/Downloads/crop-datasets.csv")
df.columns = df.columns.str.strip().str.lower()

METHOD = "mean"

if METHOD == "mean":
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
        baseline_yield = df['yield'].mean()
        baseline_area = df['area'].mean()
    if (
        pd.isna(baseline_yield) or
        pd.isna(baseline_area) or
        baseline_yield == 0 or
        baseline_area == 0
    ):
        return 0

    yield_ratio = row['yield'] / baseline_yield
    area_ratio = row['area'] / baseline_area

    score = (
        0.8 * yield_ratio +
        0.2 * area_ratio
    ) * 50

    return max(0, min(score, 100))


df['cropping_pattern_score'] = df.apply(
    calculate_score,
    axis=1
)

X = df[
    ['crop', 'crop_year', 'season',
     'state', 'area', 'yield']
]

y = df['cropping_pattern_score']

X = pd.get_dummies(
    X,
    columns=['crop', 'season', 'state'],
    drop_first=True
)

training_cols = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "crop_pattern_model.pkl")
joblib.dump(training_cols.tolist(), "training_cols.pkl")

print("Model saved successfully!")
