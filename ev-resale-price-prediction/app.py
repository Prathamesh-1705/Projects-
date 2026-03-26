from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
car_data = pd.read_csv("Electric_cars_data.csv")

# 🔥 FIX COLUMN NAMES (IMPORTANT)
car_data.columns = car_data.columns.str.strip()

# Clean data
for col in ["Brand", "Model", "Variant"]:
    car_data[col] = car_data[col].astype(str).str.strip()

car_data["Launch_year"] = pd.to_numeric(car_data["Launch_year"], errors="coerce")
car_data = car_data.dropna(subset=["Launch_year"])
car_data["Launch_year"] = car_data["Launch_year"].astype(int)

# DEBUG (optional - remove later)
print("Columns:", car_data.columns)
print("Rows:", len(car_data))


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        brands=sorted(car_data["Brand"].unique()),
        launch_years=sorted(car_data["Launch_year"].unique())
    )


# ---------------- AVAILABLE CARS ----------------
@app.route("/available-cars")
def available_cars():

    # 🔥 GROUP DATA CORRECTLY
    grouped = car_data.groupby(["Brand", "Model"])["Variant"].unique().reset_index()

    cars = []
    for _, row in grouped.iterrows():
        cars.append({
            "brand": row["Brand"],
            "model": row["Model"],
            "variants": list(row["Variant"])
        })

    print("Cars count:", len(cars))  # DEBUG

    return render_template("available_cars.html", cars=cars)


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    brand = request.form["Brand"].strip()
    model_name = request.form["Model"].strip()
    launch_year_selected = int(request.form["Launch_year"])
    variant = request.form["Variant"].strip()
    purchase_year = int(request.form["Purchase_year"])
    resale_year = int(request.form["Resale_year"])

    
    # VALIDATION

    if resale_year < purchase_year:
        return render_template(
            "index.html",
            prediction_text="🚬 Are you High On Something 🚬 ?",
            brands=sorted(car_data["Brand"].unique()),
            launch_years=sorted(car_data["Launch_year"].unique())
        )

    if purchase_year < launch_year_selected:
        return render_template(
            "index.html",
            prediction_text="❌ Purchase year cannot be earlier than car launch year.",
            brands=sorted(car_data["Brand"].unique()),
            launch_years=sorted(car_data["Launch_year"].unique())
        )

    if resale_year < launch_year_selected:
        return render_template(
            "index.html",
            prediction_text="❌ Resale year cannot be earlier than car launch year.",
            brands=sorted(car_data["Brand"].unique()),
            launch_years=sorted(car_data["Launch_year"].unique())
        )
    filtered = car_data[
        (car_data["Brand"].str.lower() == brand.lower()) &
        (car_data["Model"].str.lower() == model_name.lower()) &
        (car_data["Launch_year"] == launch_year_selected)
    ]

    if filtered.empty:
        return render_template(
            "index.html",
            prediction_text="Car not found",
            brands=sorted(car_data["Brand"].unique()),
            launch_years=sorted(car_data["Launch_year"].unique())
        )

    car = filtered[
        filtered["Variant"].str.lower() == variant.lower()
    ]

    if car.empty:
        return render_template(
            "index.html",
            prediction_text="Variant not found",
            brands=sorted(car_data["Brand"].unique()),
            launch_years=sorted(car_data["Launch_year"].unique())
        )

    car = car.iloc[0]

    launch_year = int(car["Launch_year"])
    original_price = float(car["Original_Price_INR"])
    range_km = float(car["range_km"])
    battery_capacity = float(car["Battery_Capacity_kwh"])

    car_age = resale_year - purchase_year

    yearly_data = []
    for year in range(purchase_year, resale_year + 1):
        age = year - purchase_year
        yearly_data.append({
            "year": year,
            "battery": round(battery_capacity * (0.98 ** age), 2),
            "range": round(range_km * (0.98 ** age), 2)
        })

    final_battery = battery_capacity * (0.98 ** car_age)
    final_range = range_km * (0.98 ** car_age)

    input_data = pd.DataFrame([{
        "Launch_year": launch_year,
        "Original_Price_INR": original_price,
        "range_km": range_km,
        "Purchase_year": purchase_year,
        "Resale_year": resale_year,
        "Battery_Capacity_kwh": battery_capacity,
        "Car_age": car_age,
        "Battery_Degraded": final_battery,
        "Remaining_range": final_range,
        "Brand": car["Brand"],
        "Model": car["Model"],
        "Variant": car["Variant"]
    }])

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        brands=sorted(car_data["Brand"].unique()),
        launch_years=sorted(car_data["Launch_year"].unique()),
        prediction_text=f"Estimated Resale Price: ₹ {int(prediction[0]):,}",
        launch_price=f"₹ {int(original_price):,}",
        car_age=car_age,
        yearly_data=yearly_data
    )


if __name__ == "__main__":
    app.run(debug=True)