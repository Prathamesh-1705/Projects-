import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("Electric_cars_data.csv").dropna().copy()

df['Launch_year'] = df['Launch_year'].astype(int)
df['Original_Price_INR'] = df['Original_Price_INR'].astype(float)
df['range_km'] = df['range_km'].astype(float)
df['Purchase_year'] = df['Purchase_year'].astype(int)
df['Resale_year'] = df['Resale_year'].astype(int)
df['Battery_Capacity_kwh'] = df['Battery_Capacity_kwh'].astype(float)

df['Car_age'] = df['Resale_year'] - df['Purchase_year']
df['Battery_Degraded'] = df['Battery_Capacity_kwh'] * (0.98 ** df['Car_age'])
df['Remaining_range'] = df['range_km'] * (0.98 ** df['Car_age'])

df["resale_price"] = (
    df["Original_Price_INR"] *
    (0.85 ** df["Car_age"]) *
    (df["Remaining_range"] / df["range_km"])
)

df['resale_price'] = df['resale_price'].astype(int)

X = df.drop("resale_price", axis=1)
y = df["resale_price"]

categorical_cols = ["Brand", "Model", "Variant"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

print("R2:", r2_score(y_test, pipeline.predict(X_test)))
print("MAE:", mean_absolute_error(y_test, pipeline.predict(X_test)))

pickle.dump(pipeline, open("model.pkl", "wb"))
print("✅ Model saved successfully")