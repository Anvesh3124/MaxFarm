import pandas as pd
import itertools
import re
import sys
import os
from flask import Flask, render_template, request
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)

# 🔐 Configure Gemini with your API key
genai.configure(api_key="AIzaSyDC7tJMHI8owB_33gEUEfYe1-IDAcwMwT8")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")


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
    numbers = re.findall(r"\d+", value)

    if not numbers:
        return 0

    if len(numbers) >= 2:
        return (int(numbers[0]) + int(numbers[1])) / 2

    return int(numbers[0])


def level_to_numeric(level):
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    return mapping.get(str(level).strip(), 0)


def yes_no_to_numeric(val):
    return 1 if str(val).strip().lower() == "yes" else 0


############################
# PREPROCESSING
############################

def preprocess_dataset(df):

    nutrient_cols = [
        "Nitrogen(LOW,MEDIUM,HIGH)",
        "Phosphorous(LOW,MEDIUM,HIGH)",
        "Potassium(LOW,MEDIUM,HIGH)",
        "OrganicCarbon(LOW,MEDIUM,HIGH)",
        "Sulphur(LOW,MEDIUM,HIGH)",
        "Iron(LOW,MEDIUM,HIGH)",
        "Zinc(LOW,MEDIUM,HIGH)",
        "Copper(LOW,MEDIUM,HIGH)",
        "Boron(LOW,MEDIUM,HIGH)",
        "Manganese(LOW,MEDIUM,HIGH)"
    ]

    for col in nutrient_cols:
        if col in df.columns:
            df[col] = df[col].apply(level_to_numeric)

    depletion_cols = [
        "Nitrogen Depleted",
        "Phosphorous Depleted",
        "Potassium Depleted",
        "OrganicCarbon Depleted",
        "Sulphur Depleted",
        "Iron Depleted",
        "Zinc Depleted",
        "Copper Depleted",
        "Boron Depleted",
        "Manganese Depleted"
    ]

    for col in depletion_cols:
        if col in df.columns:
            df[col] = df[col].apply(yes_no_to_numeric)

    range_cols = [
        "Water Requirement(per acre)",
        "Duration(Days)",
        "Investment Capital(per acre)",
        "Yield in rs"
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
            new_soil[soil_col] -= crop.get(soil_col, 0)
            new_soil[soil_col] = max(new_soil[soil_col], 0)

    return new_soil


def soil_penalty(soil):
    ideal = 2
    return sum((value - ideal) ** 2 for value in soil.values())


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

        total_time_used += crop["Duration(Days)"]
        total_cost += crop["Investment Capital(per acre)"]

        if total_time_used > total_time:
            return None

        if total_cost > total_budget:
            return None

        if not is_feasible(crop, soil):
            return None

        soil = update_soil(soil, crop)
        total_profit += crop["Yield in rs"]
        total_water += crop["Water Requirement(per acre)"]

    score = (
        total_profit
        - SOIL_WEIGHT * soil_penalty(soil)
        - WATER_WEIGHT * total_water
    )

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
# MAIN RUNNER
############################

def run_optimizer(csv_path, total_time, total_budget):

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = preprocess_dataset(df)

    initial_soil = {
        "Nitrogen(LOW,MEDIUM,HIGH)": 3,
        "Phosphorous(LOW,MEDIUM,HIGH)": 3,
        "Potassium(LOW,MEDIUM,HIGH)": 3,
        "OrganicCarbon(LOW,MEDIUM,HIGH)": 2,
        "Sulphur(LOW,MEDIUM,HIGH)": 2,
        "Iron(LOW,MEDIUM,HIGH)": 2,
        "Zinc(LOW,MEDIUM,HIGH)": 2,
        "Copper(LOW,MEDIUM,HIGH)": 2,
        "Boron(LOW,MEDIUM,HIGH)": 2,
        "Manganese(LOW,MEDIUM,HIGH)": 2
    }

    best = find_best_rotations(df, initial_soil, total_time, total_budget)

    print("\nTop 3 Recommended Crop Rotations:\n")

    if not best:
        print("No feasible rotation found.")
        return

    for i, result in enumerate(best):
        print(f"{i+1}. Crops: {result['crops']}")
        print(f"   Total Profit (₹): {round(result['profit'],2)}")
        print(f"   Total Initial Cost (₹): {round(result['cost'],2)}")
        print(f"   Total Duration (days): {round(result['time'],2)}\n")


############################
# ENTRY POINT
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
            "soil_type": request.form["soil_type"],
            "total_duration": int(request.form["total_duration"]),
            "investment_budget": float(request.form["investment_budget"])
        }



if __name__ == "__main__":

    if len(sys.argv) == 4:
        csv_path = sys.argv[1]
        total_time = float(sys.argv[2])
        total_budget = float(sys.argv[3])
        run_optimizer(csv_path, total_time, total_budget)

    else:
        print("Running in notebook/default mode...\n")
        run_optimizer("./crops.csv", 365, 100000)
