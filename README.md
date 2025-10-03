# 🛒 Retail Demand Forecasting for New Markets

## 📘 Project Overview  
This project focuses on **forecasting weekly demand** for a new city — **Jaipur_NewCity** — using data from other cities like **Delhi, Mumbai, Bengaluru, Kolkata, and Hyderabad**.  

The main challenge was predicting demand for a new store that doesn’t have much sales history. So, I used **transfer learning** and **machine learning models** to learn from other cities and forecast for Jaipur.

The model predicts demand for **all SKUs** for the **next 13 weeks**, along with **confidence intervals (upper & lower bounds)**.

---

## 🎯 Objective  
- Forecast weekly demand for **Jaipur_NewCity** (13 weeks ahead)  
- Include **prediction intervals** (lower and upper bounds)  
- Provide **explainability** — why the model predicted a certain value  

---

## 🧠 Approach  

### Step 1: Data Collection  
Used past sales and context data from other cities, including:  
- **Sales data** (units sold per SKU per week)  
- **Price** and **promotions**  
- **Weather** (temperature, rainfall)  
- **Holidays** and **seasonal effects**

### Step 2: Feature Engineering  
Created useful features like:  
- **Time features**: week, month, season  
- **Lag features**: previous week’s sales  
- **Rolling stats**: moving averages  
- **External features**: price, promo, holiday, weather  

### Step 3: Model Training  
Used a combination of models for better accuracy:  
- **LightGBM**  
- **XGBoost**  
- **CatBoost**  
Combined them using **weighted averaging (ensemble)**.  

### Step 4: Transfer Learning  
The model learned patterns from **existing cities** and applied them to **Jaipur**, helping forecast even without past sales data.  

---

## 📈 Results  

- The model generates **weekly forecasts** for all SKUs.  
- Each forecast includes:  
  - `forecast` (predicted units)  
  - `lower_bound` (minimum expected)  
  - `upper_bound` (maximum expected)  

Example (from `forecast.csv`):  

| week_start | market | sku_id | forecast | lower_bound | upper_bound |
|-------------|---------|--------|-----------|--------------|--------------|
| 1/20/2025 | Jaipur_NewCity | S001 | 14 | 14 | 15 |
| 1/20/2025 | Jaipur_NewCity | S002 | 14 | 14 | 15 |

---

## 🗂️ Dataset Information  

| File Name | Description |
|------------|--------------|
| `panel_train.csv` | Historical sales data from other cities |
| `weather_future.csv` | Weather forecast for Jaipur |
| `promos_future.csv` | Planned promotions |
| `price_plan_future.csv` | Future price plan |
| `calendar_future.csv` | Holiday and calendar info |
| `metadata.json` | Project configuration details |

Each file contains **weekly data** used to generate forecasts.

---

## 🧩 Tools & Libraries  
- **Python 3.11+**  
- **Pandas, NumPy** (Data Handling)  
- **LightGBM, XGBoost, CatBoost** (Modeling)  
- **Matplotlib, Seaborn, Plotly** (Visualization)  
- **SHAP** (Explainability)  

---

## ⚙️ How to Run  

```bash
# 1. Create environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the main script
python enhanced_forecasting.py
```
## 🛠️ How to Run (Basic)
1. Create and activate a Python environment:
```bash
# create venv
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
2. Install requirements:

bash
Copy code
```
pip install -r requirements.txt
```
3. Place your data files in data/ and run the main script:

bash
Copy code
```
python enhanced_forecasting.py
```
This script runs preprocessing, feature engineering, trains models, generates forecasts, and saves outputs to outputs/.

## 🧾 Quick Config (example config.yaml)
yaml
Copy code
target_city: "Jaipur_NewCity"
forecast_horizon_weeks: 13

model:
  ensemble:
    enabled: true
    models: ["lightgbm", "xgboost", "catboost"]
    weights: [0.4, 0.3, 0.3]

    
 ## 📈 Results Summary (example)
Forecasts produced for all SKUs for the next 13 weeks.

Sample forecast value (S001 @ 2025-01-20): 14 units (14–15 bounds).

Model achieved reasonable validation metrics on source cities (example):

MAE ≈ xx.x (fill with your run results)

RMSE ≈ xx.x

Explainability (reports/explainability_report.md) lists top drivers such as promo_flag, price, holiday_flag, and recent lagged sales.

(Replace xx.x with metrics from your run before submitting.)

 ## 🧪 Troubleshooting
Memory errors during feature engineering: reduce batch sizes or process SKUs in chunks.

CatBoost install fails: try pip install catboost separately or remove CatBoost from ensemble temporarily.

Date format issues: ensure week_start uses YYYY-MM-DD or MM/DD/YYYY consistently.

## 🔭 Future Improvements
Add a deep learning model (LSTM or TFT) for more complex patterns.


## 🧑‍💻 Author
Your Name = Sohamsingh Thakur
SRM UNIVERSITY
