"""
================================================================================
BATTERY PREDICTION ENGINE - USAGE GUIDE & EXAMPLES
================================================================================
How to use the prediction engine to forecast:
- Next charging time
- Delta V (voltage anomalies)
- Degradation rate / RUL (Remaining Useful Life)
- DL (Degradation Level / Days/Cycles to End of Life)
================================================================================
"""

# ================================================================================
# EXAMPLE 1: BASIC USAGE - PREDICT NEXT BATTERY STATE
# ================================================================================

from battery_prediction_engine import BatteryPredictionEngine
import pandas as pd

# Load trained models
engine = BatteryPredictionEngine('trained_battery_models')

# Current battery state
current_battery = {
    'voltage': 48.5,          # Measured voltage (V)
    'current': 5.0,           # Charging/discharging current (A)
    'temperature': 25.0,      # Battery temperature (°C)
    'capacity': 95.0,         # Remaining capacity (Ah)
    'resistance': 0.051,      # Internal resistance (Ω)
    'cycle': 100,             # Current cycle number
    'soc': 0.95,              # Current SOC (0-1)
    'soh': 98.0               # Current SOH (%)
}

# Predict next state
next_state = engine.predict_next_state(current_battery)

print("📊 NEXT STATE PREDICTION:")
print("-" * 60)
print(f"Predicted SOC:              {next_state['predictions']['next_soc']:.4f}")
print(f"Predicted SOH:              {next_state['predictions']['next_soh']:.2f}%")
print(f"Predicted ΔV:               {next_state['predictions']['delta_v']:.6f} V")
print(f"Predicted Charging Time:    {next_state['predictions']['estimated_charging_time_h']:.2f} hours")
print(f"Predicted Status:           {next_state['predictions']['predicted_status']}")
print(f"SOC Change:                 {next_state['predictions']['soc_change']:.4f}")
print(f"SOH Change:                 {next_state['predictions']['soh_change']:.2f}%")


# ================================================================================
# EXAMPLE 2: PREDICT CHARGING TIME FOR DIFFERENT SCENARIOS
# ================================================================================

"""
Predict charging time to reach 100% from different SOC levels
"""

print("\n📈 CHARGING TIME PREDICTIONS:")
print("-" * 60)

soc_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
capacity = 100  # Ah
charge_rate = 5  # A

for target_soc in soc_levels:
    charging_time = engine.predict_charging_time(
        current_soc=0.2,          # Starting from 20% SOC
        target_soc=target_soc,    # Target SOC
        capacity=capacity,
        charge_rate=charge_rate
    )
    print(f"20% → {target_soc*100:.0f}%: {charging_time:.2f} hours")

# Or from different charge rates
print("\n⚡ CHARGING TIME vs CHARGE RATE (20% → 100%):")
print("-" * 60)

charge_rates = [1.0, 5.0, 10.0, 20.0]

for rate in charge_rates:
    charging_time = engine.predict_charging_time(
        current_soc=0.2,
        target_soc=1.0,
        capacity=100,
        charge_rate=rate
    )
    print(f"Charge rate {rate:5.1f}A: {charging_time:6.2f} hours")


# ================================================================================
# EXAMPLE 3: DELTA V ANOMALY DETECTION
# ================================================================================

"""
Monitor ΔV (Delta V) to detect voltage anomalies
"""

print("\n⚠️ DELTA V ANOMALY DETECTION:")
print("-" * 60)

# Test different battery conditions
test_conditions = [
    {'name': 'Normal', 'current': 5.0, 'temp': 25.0, 'resistance': 0.051},
    {'name': 'High current', 'current': 15.0, 'temp': 25.0, 'resistance': 0.051},
    {'name': 'Hot battery', 'current': 5.0, 'temp': 40.0, 'resistance': 0.051},
    {'name': 'High resistance', 'current': 5.0, 'temp': 25.0, 'resistance': 0.100},
    {'name': 'All bad', 'current': 20.0, 'temp': 45.0, 'resistance': 0.150}
]

for condition in test_conditions:
    features = [
        48.5,                          # voltage
        condition['current'],          # current
        condition['temp'],             # temperature
        100,                           # capacity
        condition['resistance']        # resistance
    ]
    
    import numpy as np
    delta_v = engine.predict_delta_v(np.array(features))
    
    status = "🟢 NORMAL" if abs(delta_v) < 0.5 else "🔴 ANOMALY"
    print(f"{condition['name']:15} ΔV: {delta_v:+.6f} V  {status}")


# ================================================================================
# EXAMPLE 4: DEGRADATION ANALYSIS & REMAINING USEFUL LIFE (RUL)
# ================================================================================

"""
Analyze battery degradation rate and estimate remaining useful life
- DL (Degradation Level): How much SOH has degraded
- RUL (Remaining Useful Life): How many cycles/days left until EOL
"""

# Load historical data
df = pd.read_csv('example_battery_data.csv')

# Calculate degradation
degradation = engine.predict_degradation_rate(df, df.columns.tolist())

print("\n📉 DEGRADATION ANALYSIS:")
print("-" * 60)
print(f"Current SOH:                {degradation['current_soh']:.2f}%")
print(f"Degradation rate:           {degradation['degradation_rate_per_cycle']:.6f}% per cycle")
print(f"Degradation per 100 cycles: {degradation['degradation_rate_per_100_cycles']:.4f}%")
print(f"\n⏰ REMAINING USEFUL LIFE (RUL):")
print(f"Cycles to EOL (80%):        {degradation['cycles_to_eol']:.0f} cycles")
print(f"Days to EOL:                {degradation['days_to_eol']:.0f} days")
print(f"Estimated EOL date:         {degradation['estimated_eol_date']}")
print(f"Total cycles observed:      {degradation['total_cycles_observed']}")

"""
Interpretation:
- Current SOH: 98.50% - Battery is relatively new
- Degradation rate: -0.0520% per cycle
  → Battery loses ~0.052% capacity per charge cycle
- Cycles to EOL: 288 cycles
  → At current rate, battery will reach 80% SOH in 288 cycles
- Days to EOL: 288 days
  → If 1 cycle per day = ~9.6 months until replacement needed
"""


# ================================================================================
# EXAMPLE 5: FORECAST MULTIPLE STEPS AHEAD
# ================================================================================

"""
Predict battery state for the next N measurements/cycles
"""

# Predict next 20 steps
predictions_df = engine.predict_sequence(df, steps_ahead=20)

print("\n🔮 20-STEP FORECAST:")
print("-" * 60)
print(predictions_df.to_string())

"""
Output columns:
- step: Prediction step number (1-20)
- predicted_soc: State of Charge (0-1)
- predicted_soh: State of Health (%)
- predicted_delta_v: Voltage anomaly indicator (V)
- predicted_charging_time_h: Time to charge (hours)
- predicted_status: Battery status
"""


# ================================================================================
# EXAMPLE 6: COMPREHENSIVE FORECAST REPORT
# ================================================================================

"""
Generate complete forecasting report with alerts
"""

# Generate report
report = engine.generate_forecast_report(df, steps_ahead=10)

print("\n📋 COMPREHENSIVE FORECAST REPORT:")
print("=" * 80)

# Data summary
print("\n📊 Data Summary:")
for key, value in report['data_summary'].items():
    print(f"  {key}: {value}")

# Degradation
print("\n📉 Degradation Analysis:")
deg = report['degradation_analysis']
print(f"  Current SOH: {deg['current_soh']:.2f}%")
print(f"  DL (Degradation Level): {100 - deg['current_soh']:.2f}%")
print(f"  Degradation rate: {deg['degradation_rate_per_cycle']:.6f}% per cycle")
print(f"  RUL (Remaining Useful Life): {deg['cycles_to_eol']:.0f} cycles ({deg['days_to_eol']:.0f} days)")
print(f"  Estimated EOL: {deg['estimated_eol_date']}")

# Next prediction
print("\n🔮 Next Step Predictions:")
next_pred = report['next_step_prediction']['predictions']
print(f"  Next SOC: {next_pred['next_soc']:.4f}")
print(f"  Next SOH: {next_pred['next_soh']:.2f}%")
print(f"  Next ΔV: {next_pred['delta_v']:.6f} V")
print(f"  Charging time: {next_pred['estimated_charging_time_h']:.2f} hours")
print(f"  Status: {next_pred['predicted_status']}")

# Alerts
print("\n⚠️ Alerts:")
for alert in report['alerts']:
    print(f"  {alert}")

# Forecast table
print("\n📈 Forecast (10 steps):")
forecast_df = pd.DataFrame(report['forecast'])
print(forecast_df.to_string(index=False))


# ================================================================================
# EXAMPLE 7: VISUALIZE FORECAST
# ================================================================================

"""
Create visualizations of predictions
"""

# Predict next 30 steps for visualization
predictions_df = engine.predict_sequence(df, steps_ahead=30)

# Create visualization
engine.visualize_forecast(
    df,
    predictions_df,
    save_path='battery_forecast_plot.png'
)

# The plot shows:
# 1. SOC (top-left): Historical SOC + Predicted SOC trend
# 2. SOH (top-right): Historical SOH + Predicted degradation
# 3. Charging Time (bottom-left): Predicted time to full charge
# 4. Delta V (bottom-right): Voltage anomaly predictions with thresholds


# ================================================================================
# EXAMPLE 8: SAVE AND LOAD PREDICTIONS
# ================================================================================

"""
Save predictions to JSON for later analysis or integration
"""

# Generate report
report = engine.generate_forecast_report(df, steps_ahead=15)

# Save to file
engine.save_predictions(report, 'battery_predictions_20240524.json')

# Later, load and use predictions
import json

with open('battery_predictions_20240524.json', 'r') as f:
    loaded_report = json.load(f)

print("\nLoaded report keys:")
for key in loaded_report.keys():
    print(f"  - {key}")


# ================================================================================
# EXAMPLE 9: REAL-TIME PREDICTION LOOP
# ================================================================================

"""
Continuously predict battery state as new measurements arrive
"""

class RealtimeBatteryPredictor:
    """Real-time prediction system"""
    
    def __init__(self, model_dir='trained_battery_models'):
        self.engine = BatteryPredictionEngine(model_dir)
        self.measurements = []
        self.predictions = []
    
    def add_measurement(self, measurement: dict):
        """Add new measurement and predict next state"""
        
        self.measurements.append(measurement)
        
        # Predict next state
        prediction = self.engine.predict_next_state(measurement)
        self.predictions.append(prediction)
        
        # Check for alerts
        next_pred = prediction['predictions']
        if next_pred['next_soc'] < 0.2:
            print(f"⚠️  Alert: Low SOC predicted - {next_pred['next_soc']:.1%}")
        
        if abs(next_pred['delta_v']) > 0.5:
            print(f"⚠️  Alert: ΔV anomaly - {next_pred['delta_v']:.6f} V")
        
        return prediction
    
    def get_latest_prediction(self):
        """Get most recent prediction"""
        return self.predictions[-1] if self.predictions else None
    
    def get_prediction_summary(self):
        """Get summary of all predictions"""
        
        if not self.predictions:
            return "No predictions yet"
        
        summary = {
            'total_measurements': len(self.measurements),
            'total_predictions': len(self.predictions),
            'latest': self.predictions[-1],
            'soc_trend': [p['predictions']['next_soc'] for p in self.predictions[-10:]],
            'soh_trend': [p['predictions']['next_soh'] for p in self.predictions[-10:]],
            'delta_v_trend': [p['predictions']['delta_v'] for p in self.predictions[-10:]]
        }
        
        return summary


# Usage
predictor = RealtimeBatteryPredictor()

# Simulate real-time measurements
measurements = [
    {'voltage': 48.5, 'current': 5.0, 'temperature': 25.0, 'capacity': 95, 'resistance': 0.051, 'cycle': i, 'soc': 0.95-(i*0.01), 'soh': 98-(i*0.05)}
    for i in range(10)
]

print("\n⏱️ REAL-TIME PREDICTION LOOP:")
print("-" * 60)

for i, measurement in enumerate(measurements):
    pred = predictor.add_measurement(measurement)
    print(f"Step {i+1}: SOC→{pred['predictions']['next_soc']:.3f}, SOH→{pred['predictions']['next_soh']:.1f}%, ΔV→{pred['predictions']['delta_v']:+.6f}V")


# ================================================================================
# EXAMPLE 10: INTEGRATION WITH FASTAPI
# ================================================================================

"""
Deploy prediction engine as REST API
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Battery Prediction API")

class BatteryMeasurement(BaseModel):
    voltage: float
    current: float
    temperature: float
    capacity: float
    resistance: float
    cycle: int = 0
    soc: float = 0.9
    soh: float = 98.0

class PredictionResponse(BaseModel):
    timestamp: str
    next_soc: float
    next_soh: float
    delta_v: float
    charging_time_h: float
    status: str

# Global engine
prediction_engine = BatteryPredictionEngine('trained_battery_models')

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_state(measurement: BatteryMeasurement):
    """Predict next battery state"""
    
    prediction = prediction_engine.predict_next_state(measurement.dict())
    pred_data = prediction['predictions']
    
    return PredictionResponse(
        timestamp=datetime.now().isoformat(),
        next_soc=pred_data['next_soc'],
        next_soh=pred_data['next_soh'],
        delta_v=pred_data['delta_v'],
        charging_time_h=pred_data['estimated_charging_time_h'],
        status=pred_data['predicted_status']
    )

@app.get("/predict/charging-time")
async def get_charging_time(
    current_soc: float,
    target_soc: float = 1.0,
    capacity: float = 100.0,
    charge_rate: float = 5.0
):
    """Predict charging time"""
    
    charging_time = prediction_engine.predict_charging_time(
        current_soc, target_soc, capacity, charge_rate
    )
    
    return {
        'current_soc': current_soc,
        'target_soc': target_soc,
        'estimated_hours': charging_time,
        'estimated_minutes': charging_time * 60,
        'timestamp': datetime.now().isoformat()
    }

# Run with: uvicorn battery_prediction_api:app --reload


# ================================================================================
# KEY METRICS EXPLAINED
# ================================================================================

"""
📊 PREDICTION METRICS:

1. SOC (State of Charge) - Predicted
   Range: 0-1 (0% to 100%)
   Meaning: How much charge is left in the battery
   Use: Plan charging schedule

2. SOH (State of Health) - Predicted
   Range: 0-100%
   Meaning: Battery capacity vs nominal capacity
   Use: Predict end of life

3. ΔV (Delta V) - Predicted
   Meaning: Voltage difference from model estimate
   High ΔV (>0.5V): Indicates anomalies
   - Pack imbalance
   - Cell degradation
   - Connection issues
   Use: Early warning system

4. Charging Time - Predicted
   Meaning: Time needed to reach target SOC
   Depends on: Current SOC, capacity, charge rate, age
   Use: Optimize charging schedules

5. DL (Degradation Level)
   = 100% - Current SOH
   Example: SOH 95% → DL 5%
   Meaning: 5% of capacity has degraded

6. RUL (Remaining Useful Life)
   Cycles to EOL (End of Life)
   EOL threshold: Usually 80% SOH
   Use: Maintenance planning

7. Degradation Rate
   % SOH decrease per cycle
   Example: -0.05% per cycle
   = Battery loses 0.05% capacity per charge/discharge cycle
"""


print("\n✅ PREDICTION ENGINE - READY FOR USE!")
print("\nKey files:")
print("  • battery_prediction_engine.py - Main prediction engine")
print("  • This guide - Usage examples and explanations")
print("\nNext steps:")
print("  1. Train your model: python battery_model_trainer.py")
print("  2. Make predictions: python battery_prediction_engine.py")
print("  3. Or use in your code: from battery_prediction_engine import BatteryPredictionEngine")
