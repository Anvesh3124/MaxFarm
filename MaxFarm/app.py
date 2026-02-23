import pandas as pd
import re
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SOIL_WEIGHT  = 0.3
WATER_WEIGHT = 0.1

NUTRIENT_KEYS = [
    "Nitrogen", "Phosphorous", "Potassium", "OrganicCarbon",
    "Sulphur", "Iron", "Zinc", "Copper", "Boron", "Manganese",
]
NUTRIENT_COLS  = [f"{k}(LOW,MEDIUM,HIGH)" for k in NUTRIENT_KEYS]
DEPLETION_COLS = [f"{k} Depleted" for k in NUTRIENT_KEYS]

THRESHOLDS = {
    "Nitrogen":      (280,  560,  "kg/ha"),
    "Phosphorous":   (10,   25,   "kg/ha"),
    "Potassium":     (110,  280,  "kg/ha"),
    "OrganicCarbon": (0.5,  0.75, "%"),
    "Sulphur":       (10,   20,   "ppm"),
    "Iron":          (4.5,  9,    "ppm"),
    "Zinc":          (0.6,  1.2,  "ppm"),
    "Copper":        (0.2,  0.5,  "ppm"),
    "Boron":         (0.5,  1.0,  "ppm"),
    "Manganese":     (2,    4,    "ppm"),
}

# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------
def parse_range(value):
    if pd.isna(value):
        return 0.0
    value = str(value).replace("₹", "").replace(",", "").strip()
    nums = re.findall(r"\d+(?:\.\d+)?", value)
    if not nums:
        return 0.0
    return (float(nums[0]) + float(nums[1])) / 2 if len(nums) >= 2 else float(nums[0])

def level_to_num(level):
    return {"Low": 1, "Medium": 2, "High": 3}.get(str(level).strip(), 0)

def yes_no(val):
    return 1 if str(val).strip().lower() == "yes" else 0

# ---------------------------------------------------------------------------
# SOIL CLASSIFICATION
# ---------------------------------------------------------------------------
def classify_nutrient(key, value):
    low, high, _ = THRESHOLDS[key]
    if value < low:   return "Low"
    if value <= high: return "Medium"
    return "High"

def classify_ph(ph):
    if ph < 6.5:  return "Acidic"
    if ph <= 7.5: return "Neutral"
    return "Alkaline"

def classify_ec(ec):
    return "Non-Saline" if ec < 1.0 else "Saline"

def build_soil_profile(user_input):
    profile = {k: classify_nutrient(k, user_input[k]) for k in THRESHOLDS}
    profile["pH"] = classify_ph(user_input["pH"])
    profile["EC"]  = classify_ec(user_input["EC"])
    return profile

def build_numeric_soil(user_input):
    return tuple(
        level_to_num(classify_nutrient(k, user_input[k]))
        for k in NUTRIENT_KEYS
    )

# ---------------------------------------------------------------------------
# DATASET PREPROCESSING
# ---------------------------------------------------------------------------
def load_and_preprocess():
    path = os.path.join(os.path.dirname(__file__), "Crop_Data.csv")
    df = pd.read_csv(path)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(level_to_num)
    for col in DEPLETION_COLS:
        if col in df.columns:
            df[col] = df[col].apply(yes_no)
    for col in ["Water Requirement(per acre)", "Duration(Days)",
                "Investment Capital(per acre)", "Yield in rs"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_range)
    return df.to_dict(orient="records")

# ---------------------------------------------------------------------------
# SOIL STATE HELPERS
# ---------------------------------------------------------------------------
def is_feasible(crop, soil):
    for i, col in enumerate(NUTRIENT_COLS):
        if soil[i] < crop.get(col, 0):
            return False
    return True

def apply_crop(soil, crop):
    soil = list(soil)
    for i, (dep_col, soil_col) in enumerate(zip(DEPLETION_COLS, NUTRIENT_COLS)):
        if crop.get(dep_col, 0) == 1:
            soil[i] = max(0.0, soil[i] - crop.get(soil_col, 1) * 0.5)
    return tuple(soil)

def soil_penalty(soil):
    return sum((2 - v) ** 2 for v in soil if v < 2)

def final_score(profit, water, soil):
    return profit - SOIL_WEIGHT * soil_penalty(soil) - WATER_WEIGHT * water

# ---------------------------------------------------------------------------
# DYNAMIC MAX DEPTH
# Compute the maximum number of crops that could realistically fit
# given the shortest/cheapest crops available and the user's constraints.
# Capped at 10 to keep search tractable.
# ---------------------------------------------------------------------------
def compute_max_depth(crops, total_days, total_budget):
    min_days   = min((c.get("Duration(Days)", 9999)            for c in crops), default=1)
    min_budget = min((c.get("Investment Capital(per acre)", 9999) for c in crops), default=1)
    depth_by_time   = int(total_days   // max(min_days,   1))
    depth_by_budget = int(total_budget // max(min_budget, 1))
    return min(depth_by_time, depth_by_budget, 10)   # hard cap at 10

# ---------------------------------------------------------------------------
# BRANCH & BOUND
# ---------------------------------------------------------------------------
def find_best_rotations(crops, initial_soil, total_days, total_budget, top_k=3):
    """
    Branch & Bound with dynamic depth.

    Key fixes vs previous version:
    1. MAX_ROTATION_LENGTH is now computed from actual time+budget, not hardcoded.
    2. Upper bound uses `slots_left` relative to the CURRENT node's remaining
       capacity, not a fixed global depth — so it stays tight at every level.
    3. The `break` optimisation is kept (crops sorted DESC by yield), but only
       fires when the relaxed UB genuinely cannot beat the worst top-k score.
    """
    max_depth      = compute_max_depth(crops, total_days, total_budget)
    crops_by_yield = sorted(crops, key=lambda c: c.get("Yield in rs", 0), reverse=True)

    # Precompute sorted yields once for fast upper-bound calculation
    all_yields_desc = sorted(
        [(c.get("Yield in rs", 0), c.get("Duration(Days)", 0),
          c.get("Investment Capital(per acre)", 0))
         for c in crops],
        key=lambda x: x[0], reverse=True
    )

    results = []   # kept sorted score ASC; [0] = worst of best

    def worst_score():
        return results[0]["score"] if len(results) == top_k else float("-inf")

    def upper_bound(days_left, budget_left, current_profit):
        """
        Relaxed bound: greedily fill remaining time+budget with the
        highest-yield crops (no soil-feasibility check — true upper bound).
        Stops when neither time nor budget allows another crop.
        """
        profit = current_profit
        d_left = days_left
        b_left = budget_left
        for yield_val, dur, cost in all_yields_desc:
            if d_left <= 0 or b_left <= 0:
                break
            if dur <= d_left and cost <= b_left:
                profit += yield_val
                d_left -= dur
                b_left -= cost
        return profit

    def branch(sequence, soil, days_used, budget_used, profit, water):
        depth     = len(sequence)
        days_left   = total_days   - days_used
        budget_left = total_budget - budget_used

        # Record every non-empty valid sequence as a candidate result
        if depth > 0:
            score = final_score(profit, water, soil)
            if score > worst_score():
                results.append({
                    "crops":        [c["Crop Name"] for c in sequence],
                    "crop_records": list(sequence),
                    "profit":       profit,
                    "cost":         budget_used,
                    "time":         days_used,
                    "water":        water,
                    "score":        score,
                    "soil_end":     soil,
                })
                results.sort(key=lambda x: x["score"])
                if len(results) > top_k:
                    results.pop(0)

        # Stop when depth limit or no resources left
        if depth >= max_depth or days_left <= 0 or budget_left <= 0:
            return

        for crop in crops_by_yield:
            cd = crop.get("Duration(Days)", 0)
            cb = crop.get("Investment Capital(per acre)", 0)

            if days_used + cd > total_days:     continue
            if budget_used + cb > total_budget: continue
            if not is_feasible(crop, soil):     continue

            # Prune: best possible profit from this node onwards
            ub = upper_bound(
                days_left  - cd,
                budget_left - cb,
                profit + crop.get("Yield in rs", 0),
            )
            if ub <= worst_score():
                # Crops sorted DESC — remaining crops can only be worse, break
                break

            branch(
                sequence + [crop],
                apply_crop(soil, crop),
                days_used   + cd,
                budget_used + cb,
                profit      + crop.get("Yield in rs", 0),
                water       + crop.get("Water Requirement(per acre)", 0),
            )

    branch([], initial_soil, 0, 0, 0, 0)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ---------------------------------------------------------------------------
# BUILD NUTRIENT LOG
# ---------------------------------------------------------------------------
def build_nutrient_log(crop_records):
    log = []
    for crop in crop_records:
        depleted = [
            NUTRIENT_KEYS[i]
            for i, dep_col in enumerate(DEPLETION_COLS)
            if crop.get(dep_col, 0) == 1
        ]
        changes = ", ".join(f"{n} -10%" for n in depleted) if depleted else "No significant depletion"
        log.append(f"After {crop['Crop Name']}: {changes}")
    return log

# ---------------------------------------------------------------------------
# FORMAT RESULTS
# ---------------------------------------------------------------------------
def format_results(raw_results):
    formatted = []
    for i, r in enumerate(raw_results):
        months    = round(r["time"] / 30, 1)
        durations = " → ".join(
            f"{c['Crop Name']} ({int(c.get('Duration(Days)', 0))} days)"
            for c in r["crop_records"]
        )
        nutrient_lines = build_nutrient_log(r["crop_records"])
        soil_pen = round(SOIL_WEIGHT * soil_penalty(r["soil_end"]), 2)

        formatted.append({
            "id":       i + 1,
            "sequence": " → ".join(r["crops"]),
            "duration": f"{durations}  (Total: ~{months} months)",
            "investment": f"₹{int(r['cost']):,}",
            "yield_profit": f"₹{int(r['profit']):,} estimated profit  |  Water: {int(r['water']):,} L/acre",
            "nutrient_tracking": nutrient_lines,
            "reasoning": (
                f"Branch & Bound optimal — {len(r['crops'])} crop(s) across {months} months "
                f"(score: {round(r['score'], 1)}).\n"
                f"Total investment ₹{int(r['cost']):,} yields ₹{int(r['profit']):,} profit.\n"
                f"Soil health deduction: {soil_pen}  |  "
                f"Water cost deduction: {round(WATER_WEIGHT * r['water'], 2)}.\n"
                f"Every crop verified feasible against soil nutrient levels before selection."
            ),
        })
    return formatted

# ---------------------------------------------------------------------------
# FLASK ROUTE
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result       = None
    error        = None
    soil_profile = None

    if request.method == "POST":
        try:
            user_input = {
                "Nitrogen":          float(request.form["Nitrogen"]),
                "Phosphorous":       float(request.form["Phosphorous"]),
                "Potassium":         float(request.form["Potassium"]),
                "OrganicCarbon":     float(request.form["OrganicCarbon"]),
                "pH":                float(request.form["pH"]),
                "EC":                float(request.form["EC"]),
                "Sulphur":           float(request.form["Sulphur"]),
                "Iron":              float(request.form["Iron"]),
                "Zinc":              float(request.form["Zinc"]),
                "Copper":            float(request.form["Copper"]),
                "Boron":             float(request.form["Boron"]),
                "Manganese":         float(request.form["Manganese"]),
                "soil_type":         request.form.get("soil_type", "").strip(),
                "total_duration":    float(request.form["total_duration"]),
                "investment_budget": float(request.form["investment_budget"]),
            }

            soil_profile = build_soil_profile(user_input)
            initial_soil = build_numeric_soil(user_input)
            crops        = load_and_preprocess()
            total_days   = user_input["total_duration"] * 30
            total_budget = user_input["investment_budget"]

            raw = find_best_rotations(crops, initial_soil, total_days, total_budget)

            if not raw:
                error = ("No valid crop rotations found within your budget and duration. "
                         "Try increasing your budget or duration.")
            else:
                result = format_results(raw)

        except KeyError as ke:
            error = f"Missing form field: {ke}. Please fill in all fields."
        except ValueError as ve:
            error = f"Invalid value: {ve}"
        except Exception as e:
            error = f"Unexpected error: {e}"

    return render_template("index.html",
                           result=result,
                           error=error,
                           soil_profile=soil_profile)

if __name__ == "__main__":
    app.run(debug=True, port=5000)