import pandas as pd
import itertools
import re
import sys
import os
from flask import Flask, render_template, request

app = Flask(__name__)

############################
# CONFIGURATION
############################

MAX_ROTATION_LENGTH = 3
SOIL_WEIGHT = 0.3
WATER_WEIGHT = 0.1

############################
# UTILITY FUNCTIONS
############################

def parse_range(value):
    if pd.isna(value):
        return 0
    value = str(value).replace("₹", "").replace(",", "").strip()
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    if not numbers:
        return 0
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    return float(numbers[0])


def level_to_numeric(level):
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(str(level).strip(), 0)

def yes_no_to_numeric(val):
    return 1 if str(val).strip().lower() == "yes" else 0

############################
# SOIL CLASSIFICATION
############################

def classify_value(value, low, high):
    if value < low:
        return "Low"
    elif low <= value <= high:
        return "Medium"
    else:
        return "High"

def build_initial_soil_from_user(user_input):
    classified = {
        "Nitrogen(LOW,MEDIUM,HIGH)": classify_value(user_input["Nitrogen"], 280, 560),
        "Phosphorous(LOW,MEDIUM,HIGH)": classify_value(user_input["Phosphorous"], 10, 25),
        "Potassium(LOW,MEDIUM,HIGH)": classify_value(user_input["Potassium"], 110, 280),
        "OrganicCarbon(LOW,MEDIUM,HIGH)": classify_value(user_input["OrganicCarbon"], 0.5, 0.75),
        "Sulphur(LOW,MEDIUM,HIGH)": classify_value(user_input["Sulphur"], 10, 20),
        "Iron(LOW,MEDIUM,HIGH)": classify_value(user_input["Iron"], 4.5, 9),
        "Zinc(LOW,MEDIUM,HIGH)": classify_value(user_input["Zinc"], 0.6, 1.2),
        "Copper(LOW,MEDIUM,HIGH)": classify_value(user_input["Copper"], 0.2, 0.5),
        "Boron(LOW,MEDIUM,HIGH)": classify_value(user_input["Boron"], 0.5, 1.0),
        "Manganese(LOW,MEDIUM,HIGH)": classify_value(user_input["Manganese"], 2, 4),
    }
    numeric_soil = {k: level_to_numeric(v) for k, v in classified.items()}
    return numeric_soil

############################
# PREPROCESS DATASET
############################

def preprocess_dataset(df):
    nutrient_cols = [
        "Nitrogen(LOW,MEDIUM,HIGH)", "Phosphorous(LOW,MEDIUM,HIGH)", "Potassium(LOW,MEDIUM,HIGH)",
        "OrganicCarbon(LOW,MEDIUM,HIGH)", "Sulphur(LOW,MEDIUM,HIGH)", "Iron(LOW,MEDIUM,HIGH)",
        "Zinc(LOW,MEDIUM,HIGH)", "Copper(LOW,MEDIUM,HIGH)", "Boron(LOW,MEDIUM,HIGH)", "Manganese(LOW,MEDIUM,HIGH)"
    ]
    for col in nutrient_cols:
        if col in df.columns:
            df[col] = df[col].apply(level_to_numeric)

    depletion_cols = [
        "Nitrogen Depleted", "Phosphorous Depleted", "Potassium Depleted", "OrganicCarbon Depleted",
        "Sulphur Depleted", "Iron Depleted", "Zinc Depleted", "Copper Depleted", "Boron Depleted", "Manganese Depleted"
    ]
    for col in depletion_cols:
        if col in df.columns:
            df[col] = df[col].apply(yes_no_to_numeric)

    range_cols = [
        "Water Requirement(per acre)", "Duration(Days)", "Investment Capital(per acre)", "Yield in rs"
    ]
    for col in range_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_range)
    return df

############################
# SOIL MODEL
############################

def is_feasible(crop, soil):
    for nutrient in soil:
        if soil[nutrient] < crop.get(nutrient, 0):
            # print(f"{crop['Crop Name']} not feasible due to {nutrient}")
            return False
    return True

def update_soil(soil, crop):
    depletion_map = {
        "Nitrogen Depleted": "Nitrogen(LOW,MEDIUM,HIGH)",
        "Phosphorous Depleted": "Phosphorous(LOW,MEDIUM,HIGH)",
        "Potassium Depleted": "Potassium(LOW,MEDIUM,HIGH)",
        "OrganicCarbon Depleted": "OrganicCarbon(LOW,MEDIUM,HIGH)",
        "Sulphur Depleted": "Sulphur(LOW,MEDIUM,HIGH)",
        "Iron Depleted": "Iron(LOW,MEDIUM,HIGH)",
        "Zinc Depleted": "Zinc(LOW,MEDIUM,HIGH)",
        "Copper Depleted": "Copper(LOW,MEDIUM,HIGH)",
        "Boron Depleted": "Boron(LOW,MEDIUM,HIGH)",
        "Manganese Depleted": "Manganese(LOW,MEDIUM,HIGH)"
    }
    new_soil = soil.copy()
    for dep_col, soil_col in depletion_map.items():
        if crop.get(dep_col, 0) == 1:
            new_soil[soil_col] -= crop.get(soil_col, 1) * 0.5  # reduce by half the crop level
            new_soil[soil_col] = max(new_soil[soil_col], 0)
    return new_soil

def soil_penalty(soil):
    penalty = 0
    for nutrient, value in soil.items():
        if value < 2:  # Only penalize if nutrient is below medium
            penalty += (2 - value) ** 2
    return penalty
############################
# ROTATION SIMULATION
############################

def simulate_rotation(sequence, initial_soil, total_time, total_budget):
    soil = initial_soil.copy()
    total_time_used = 0
    total_cost = 0
    total_profit = 0
    total_water = 0

    for crop in sequence:
        total_time_used += crop.get("Duration(Days)", 0)
        total_cost += crop.get("Investment Capital(per acre)", 0)
        if total_time_used > total_time or total_cost > total_budget:
            return None
        if not is_feasible(crop, soil):
            return None
        soil = update_soil(soil, crop)
        total_profit += crop.get("Yield in rs", 0)
        total_water += crop.get("Water Requirement(per acre)", 0)

    score = total_profit - SOIL_WEIGHT * soil_penalty(soil) - WATER_WEIGHT * total_water

    return {
        "crops": [c["Crop Name"] for c in sequence],
        "profit": total_profit,
        "cost": total_cost,
        "time": total_time_used,
        "score": score
    }

############################
# OPTIMIZER
############################

def find_best_rotations(df, initial_soil, total_time, total_budget, top_k=3):
    crops = df.to_dict(orient="records")
    results = []
    for length in range(1, MAX_ROTATION_LENGTH + 1):
        for sequence in itertools.product(crops, repeat=length):
            result = simulate_rotation(sequence, initial_soil, total_time, total_budget)
            if result:
                results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

############################
# FLASK ROUTE
############################

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = {
            "Nitrogen": float(request.form["Nitrogen"]),
            "Phosphorous": float(request.form["Phosphorous"]),
            "Potassium": float(request.form["Potassium"]),
            "OrganicCarbon": float(request.form["OrganicCarbon"]),
            "pH": float(request.form["pH"]),
            "EC": float(request.form["EC"]),
            "Sulphur": float(request.form["Sulphur"]),
            "Iron": float(request.form["Iron"]),
            "Zinc": float(request.form["Zinc"]),
            "Copper": float(request.form["Copper"]),
            "Boron": float(request.form["Boron"]),
            "Manganese": float(request.form["Manganese"]),
        }

        total_time_months = float(request.form["total_duration"])
        total_time_days = total_time_months * 30  # Convert months → days
        total_budget = float(request.form["investment_budget"])

        df_local = pd.read_csv("Crop_Data.csv")
        df_local = preprocess_dataset(df_local)

        initial_soil = build_initial_soil_from_user(user_input)

        result = find_best_rotations(df_local, initial_soil, total_time_days, total_budget)

        print("DEBUG - Rotations Found:", len(result))
        for r in result:
            print(r)

    return render_template("index.html", result=result)

############################
# TERMINAL MODE SUPPORT
############################

if __name__ == "__main__":
    if len(sys.argv) == 4:
        csv_path = sys.argv[1]
        total_time = float(sys.argv[2])
        total_budget = float(sys.argv[3])

        df = pd.read_csv(csv_path)
        df = preprocess_dataset(df)

        initial_soil = {k: 2 for k in [
            "Nitrogen(LOW,MEDIUM,HIGH)", "Phosphorous(LOW,MEDIUM,HIGH)", "Potassium(LOW,MEDIUM,HIGH)",
            "OrganicCarbon(LOW,MEDIUM,HIGH)", "Sulphur(LOW,MEDIUM,HIGH)", "Iron(LOW,MEDIUM,HIGH)",
            "Zinc(LOW,MEDIUM,HIGH)", "Copper(LOW,MEDIUM,HIGH)", "Boron(LOW,MEDIUM,HIGH)", "Manganese(LOW,MEDIUM,HIGH)"
        ]}

        best = find_best_rotations(df, initial_soil, total_time, total_budget)
        print("\nTop 3 Recommended Crop Rotations:\n")
        for i, r in enumerate(best):
            print(f"{i+1}. Crops: {r['crops']}")
            print(f"   Profit: ₹{r['profit']}")
            print(f"   Cost: ₹{r['cost']}")
            print(f"   Duration: {r['time']} days\n")
    else:
        app.run(debug=True)
