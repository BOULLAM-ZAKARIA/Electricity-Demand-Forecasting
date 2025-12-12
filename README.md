# 📊 Electricity Demand Forecasting (Time-Series ML Project)

This project builds a machine learning model to **forecast hourly electricity demand** using 5 years of historical weather and consumption data.  
The workflow covers the full data science lifecycle: cleaning, feature engineering, exploratory analysis, modeling with XGBoost, evaluation, and insights.

---

# 1️⃣ Project Objective

To predict **future hourly electricity demand** using:

- Historical demand patterns  
- Weather conditions (Temperature, Humidity)  
- Time-based seasonal effects (hour, day, month, etc.)  

This type of forecasting is essential for:

- Energy utilities  
- Power grid optimization  
- Capacity planning  
- Smart city operations

---

# 2️⃣ Dataset Description

The dataset contains **5 years of hourly data**, with the following columns:

- **DateTime** → timestamp  
- **Temperature** → hourly temperature (°C)  
- **Humidity** → hourly humidity (%)  
- **Demand** → electricity consumption (MW)

A total of ~43,000 hourly observations.

---

# 3️⃣ Data Preprocessing Steps

1. **Parsed DateTime** and created a proper time index  
2. **Handled missing values** using:
   - `interpolate(method="linear")`  
   - forward/backward fill for consistency  
3. Verified dataset continuity (hourly frequency)  
4. Converted all relevant features to numeric formats  

Outcome → A clean, continuous time-series ready for feature engineering.

---

# 4️⃣ Feature Engineering (Critical Step)

To capture seasonality, weather patterns, and autocorrelation, the following features were created:

### 📅 4.1 Time-Based Features
These help the model learn seasonal patterns:

- `hour`  
- `dayofweek`  
- `month`  
- `year`  
- `quarter`  
- `weekofyear`  
- `dayofyear`  
- `is_weekend`

### 📈 4.2 Lag Features
These capture **autocorrelation** in electricity demand:

- `Demand_lag_24h` → demand at same hour yesterday  
- `Demand_lag_168h` → demand at same hour last week  

### 🧮 4.3 Rolling Window Statistics
Capture recent behavior trends:

- `rolling_mean_24h`  
- `rolling_std_24h`  

These features significantly improved model accuracy.

---

# 5️⃣ Train/Test Split (Time-Series Aware)

A chronological split was used:

- **Train:** first 80% of data  
- **Test:** last 20% (future unseen data)

No shuffling is allowed in forecasting tasks.

---

# 6️⃣ Modeling — XGBoost Regressor

XGBoost was selected because it handles:

- Non-linear patterns  
- Strong feature interactions  
- Seasonality via engineered features  
- Overfitting control via early stopping  

### Model configuration:

```python
model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    random_state=42
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

7️⃣ Evaluation Metrics

The model was evaluated using standard forecasting metrics:

RMSE → penalizes large errors (important for energy forecasting)

MAE → robust to outliers

RMSE: 173.68
MAE: 123.08


