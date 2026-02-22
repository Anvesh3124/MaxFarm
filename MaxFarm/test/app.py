from flask import Flask, render_template, request
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)

# 🔐 Configure Gemini with your API key
genai.configure(api_key="AIzaSyDC7tJMHI8owB_33gEUEfYe1-IDAcwMwT8")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")

# 📄 Load the dataset (this must be in the same folder)
df = pd.read_csv("Crop_Data.csv")
csv_content = df.to_string(index=False)

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

        # 🧪 Few-shot example
        example = """
EXAMPLE OUTPUT

Crop Cycle 1:
1. Crop Sequence: [Wheat → Maize → Chickpea]
2. Duration: 4 months → 3 months → 3 months (Total ~10 months)
3. Investment: ₹24,000
4. Yield & Profit: 6200 kg total, ₹58,000 profit
5. Nutrient Tracking:
   - After Wheat: Nitrogen -10%, Phosphorous -10%
   - After Maize: Nitrogen -10%, Potassium -10%
   - After Chickpea: Organic Carbon + (natural enrichment)
6. Reasoning: Wheat is nutrient-light and grows well in neutral pH. Maize adds biomass. Chickpea restores nitrogen and balances the cycle. Total profit and efficiency are high.

"""

        # 🧠 Full Gemini prompt
        prompt = f"""
You are **AgriBot**, an intelligent agricultural planner that uses Generative AI to optimize crop cycles based on soil nutrient levels, duration, and budget.

Your job is to recommend the top **3 optimized crop cycles** using:
- Document understanding of the dataset
- Few-shot prompting
- Structured and grounded generation

Each crop cycle recommendation must include:
1. Crop Sequence: [Crop A → Crop B → Crop C]
2. Duration per crop and total cycle length
3. Total Investment required (₹)
4. Estimated Yield (in kg) and Profit (in ₹)
5. Nutrient impact after each crop (10% depletion for nutrients used)
6. Final recommendation rationale (e.g., soil match, profit, nutrient balance)

Use the following thresholds for nutrient levels:
Nitrogen (kg/ha): LOW <280, MEDIUM 280–560, HIGH >560  
Phosphorous (kg/ha): LOW <10, MEDIUM 10–25, HIGH >25  
Potassium (kg/ha): LOW <110, MEDIUM 110–280, HIGH >280  
Organic Carbon (%): LOW <0.5, MEDIUM 0.5–0.75, HIGH >0.75  
pH: Acidic <6.5, Neutral 6.5–7.5, Alkaline >7.5  
EC (dS/m): Non-saline <1.0, Saline >1.0  
Sulphur (ppm): LOW <10, MEDIUM 10–20, HIGH >20  
Iron (ppm): LOW <4.5, MEDIUM 4.5–9, HIGH >9  
Zinc (ppm): LOW <0.6, MEDIUM 0.6–1.2, HIGH >1.2  
Copper (ppm): LOW <0.2, MEDIUM 0.2–0.5, HIGH >0.5  
Boron (ppm): LOW <0.5, MEDIUM 0.5–1.0, HIGH >1.0  
Manganese (ppm): LOW <2, MEDIUM 2–4, HIGH >4

🧾 Dataset Info:  
This dataset includes crop names, soil compatibility, duration, cost, expected yield, nutrient impact, and profitability.

📄 CROP DATASET:  
{csv_content}

🌱 USER FIELD PROFILE:  
{user_input}

🔁 Depletion Logic:  
After each crop, reduce the used nutrients by 10% before selecting the next crop.

⚠️ Constraints:
- Stay within budget and time limit
- Only suggest crops compatible with the user's soil type
- Maximize profit while keeping soil balanced

Now return 3 optimized crop cycles in the format shown above. Do not return code or implementation logic.
"""

        final_prompt = example + prompt
        response = model.generate_content(final_prompt)
        result = response.text

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
