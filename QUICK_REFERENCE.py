"""
═══════════════════════════════════════════════════════════════════════════════
DIGITAL TWIN - QUICK REFERENCE & CODE SNIPPETS
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC USAGE - MINIMAL SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Quick prediction using pre-trained models
from bms_inference import LightweightDigitalTwin

dt = LightweightDigitalTwin('models')

result = dt.predict(
    current=5.0,
    voltage=48.5,
    temperature=25.0,
    capacity=95.0,
    resistance=0.051
)

print(f"SOC: {result['SOC']:.2%}")
print(f"SOH: {result['SOH']:.1f}%")
print(f"Status: {result['Status']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FULL PIPELINE - TRAIN FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

from digital_twin_pipeline import (
    DataGenerator, DataPreprocessor,
    SOCModel, SOHModel, ChargingTimePredictor, DeltaVModel,
    DigitalTwin
)

# Generate/load data
df_time, df_cycle, df_step, df_test = DataGenerator.generate_test_data()

# Preprocess
preprocessor = DataPreprocessor()
features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
df_time = preprocessor.preprocess_time_series(df_time, features)
df_time = preprocessor.create_soc_from_capacity(df_time)
df_cycle = preprocessor.create_soh_from_cycles(df_cycle)

# Train models
soc_model = SOCModel(sequence_length=20, features_dim=len(features))
soc_model.train(df_time, features, epochs=10)

soh_model = SOHModel()
soh_model.train(df_cycle)

charge_model = ChargingTimePredictor()
charge_model.train(df_step)

dv_model = DeltaVModel()
dv_model.train(df_time)

# Create Digital Twin
digital_twin = DigitalTwin(soc_model, soh_model, charge_model, dv_model, preprocessor)

# Make predictions
result = digital_twin.predict_step({'Current': 5, 'Voltage': 48.5, 'Temperature': 25, 'Capacity': 95, 'Resistance': 0.051})
print(result)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BATCH PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np

dt = LightweightDigitalTwin('models')

# Generate multiple test cases
results = []
for i in range(100):
    result = dt.predict(
        current=5 * np.sin(i/50),
        voltage=48 + np.cos(i/50),
        temperature=25 + 5 * np.sin(i/70),
        capacity=100 - i*0.3,
        resistance=0.05 + i*0.00001
    )
    results.append(result)

# Convert to DataFrame
import pandas as pd
df_results = pd.DataFrame(results)
print(df_results[['SOC', 'SOH', 'Status']].describe())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. API USAGE - REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "Current": 5.0,
        "Voltage": 48.5,
        "Temperature": 25.0,
        "Capacity": 95.0,
        "Resistance": 0.051
    }
)
prediction = response.json()
print(prediction)

# Batch predictions
batch_data = [
    {"Current": 5, "Voltage": 48, "Temperature": 25, "Capacity": 95, "Resistance": 0.051},
    {"Current": 3, "Voltage": 47, "Temperature": 28, "Capacity": 80, "Resistance": 0.052},
]
response = requests.post(f"{BASE_URL}/batch", json=batch_data)
results = response.json()

# Get statistics
response = requests.get(f"{BASE_URL}/stats")
stats = response.json()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Avg SOC: {stats['avg_SOC']:.2%}")

# Generate health report
response = requests.post(
    f"{BASE_URL}/report/BATTERY_001",
    json={
        "Current": 5.0,
        "Voltage": 48.5,
        "Temperature": 25.0,
        "Capacity": 95.0,
        "Resistance": 0.051
    }
)
report = response.json()
print(report)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BMS HARDWARE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

from bms_inference import BMSInterface, LightweightDigitalTwin

dt = LightweightDigitalTwin('models')
bms = BMSInterface(dt)

# Receive measurement from hardware
result = bms.receive_measurement(
    current=5.0,
    voltage=48.5,
    temperature=25.0,
    capacity=95.0,
    resistance=0.051
)

# Get formatted display output
display = bms.format_for_display(result)
print(display)

# Get control command
command = bms.get_command(result)
print(f"Control command: {command}")

# Possible commands:
# - EMERGENCY_STOP (critical SOC)
# - START_CHARGING (low SOC)
# - REDUCE_DISCHARGE_RATE (degraded SOH)
# - CHECK_CELLS (anomaly detected)
# - NORMAL_OPERATION (all good)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DATA PREPROCESSING CUSTOM
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load your data
df = pd.read_csv('battery_data.csv')

# Remove NaN values
df = df.dropna()

# Remove outliers (IQR method)
Q1 = df['Voltage'].quantile(0.25)
Q3 = df['Voltage'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Voltage'] >= Q1 - 3*IQR) & (df['Voltage'] <= Q3 + 3*IQR)]

# Normalize features
scaler = MinMaxScaler()
features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
df[features] = scaler.fit_transform(df[features])

# Create SOC from capacity
df['SOC'] = df['Capacity'] / df['Capacity'].max()

print(df.head())


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE & LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

from digital_twin_pipeline import SOCModel, SOHModel
import joblib

# Save SOC LSTM model
soc_model = SOCModel()
soc_model.save('models/my_soc_model.h5')

# Save SOH XGBoost model
soh_model = SOHModel()
soh_model.save('models/my_soh_model.json')

# Save custom models
joblib.dump(custom_model, 'models/my_custom_model.pkl')

# Load models
soc_loaded = SOCModel()
soc_loaded.load('models/my_soc_model.h5')

soh_loaded = SOHModel()
soh_loaded.load('models/my_soh_model.json')

custom_loaded = joblib.load('models/my_custom_model.pkl')


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CUSTOM DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd

# Load Excel data
excel_file = "battery_test.xlsx"

df_time = pd.read_excel(excel_file, sheet_name=0)
df_test = pd.read_excel(excel_file, sheet_name=1)
df_cycle = pd.read_excel(excel_file, sheet_name=2)
df_step = pd.read_excel(excel_file, sheet_name=3)

# Or load CSV
df_time = pd.read_csv('time_series.csv')
df_cycle = pd.read_csv('cycle_data.csv')

# Print shape info
print(f"Time series: {df_time.shape}")
print(f"Columns: {df_time.columns.tolist()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALIZATION & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import pandas as pd

# Plot predictions
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(results_df['SOC'])
plt.xlabel('Time Step')
plt.ylabel('SOC')
plt.title('State of Charge')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(results_df['SOH'])
plt.xlabel('Time Step')
plt.ylabel('SOH (%)')
plt.title('State of Health')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(results_df['Delta_V'])
plt.xlabel('Time Step')
plt.ylabel('ΔV (V)')
plt.title('Voltage Residual')
plt.grid(True)

plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ERROR HANDLING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

from bms_inference import LightweightDigitalTwin
import numpy as np

dt = LightweightDigitalTwin('models')

# Validate input data
def validate_input(current, voltage, temperature, capacity, resistance):
    """Validate input ranges"""
    
    if not (-50 <= current <= 50):
        print("⚠️ Warning: Unusual current value")
    
    if not (30 <= voltage <= 60):
        print("⚠️ Warning: Unusual voltage value")
    
    if not (-20 <= temperature <= 60):
        print("⚠️ Warning: Unusual temperature value")
    
    if not (0 <= capacity <= 150):
        print("⚠️ Warning: Unusual capacity value")
    
    if not (0.01 <= resistance <= 0.5):
        print("⚠️ Warning: Unusual resistance value")
    
    return True

# Safe prediction with error handling
try:
    validate_input(5, 48.5, 25, 95, 0.051)
    result = dt.predict(5, 48.5, 25, 95, 0.051)
    print(result)
except Exception as e:
    print(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PERFORMANCE MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

import time
import numpy as np

dt = LightweightDigitalTwin('models')

# Time multiple predictions
times = []
for i in range(100):
    start = time.time()
    result = dt.predict(5, 48, 25, 95, 0.051)
    elapsed = time.time() - start
    times.append(elapsed * 1000)  # Convert to ms

avg_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)

print(f"Inference time statistics:")
print(f"  Average: {avg_time:.2f}ms")
print(f"  Std Dev: {std_time:.2f}ms")
print(f"  Min: {min_time:.2f}ms")
print(f"  Max: {max_time:.2f}ms")
print(f"  Throughput: {1000/avg_time:.0f} predictions/sec")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

from digital_twin_pipeline import SOCModel, SOHModel, ChargingTimePredictor, DeltaVModel

# SOC Model hyperparameters
soc_model = SOCModel(sequence_length=20, features_dim=5)
X_val, y_val = soc_model.train(
    df_time,
    features,
    epochs=20,           # More epochs for better fit
    batch_size=32,       # Smaller batch for stability
    validation_split=0.2
)

# SOH Model hyperparameters
soh_model = SOHModel()
# Internally uses:
# - n_estimators=100
# - max_depth=6
# - learning_rate=0.1
# - subsample=0.8
# - colsample_bytree=0.8

# Charging Model hyperparameters
charge_model = ChargingTimePredictor()
# Internally uses:
# - n_estimators=100
# - max_depth=10

# ΔV Model hyperparameters
dv_model = DeltaVModel()
# Internally uses:
# - n_estimators=100
# - max_depth=5
# - learning_rate=0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DEPLOYMENT CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

"""
BEFORE DEPLOYING TO PRODUCTION:

Data:
  ✓ Collected sufficient training data (>1000 samples)
  ✓ Data quality validated (cleaned, normalized)
  ✓ Test set performance acceptable
  ✓ Data privacy/security ensured

Models:
  ✓ All models trained and validated
  ✓ Performance metrics acceptable
  ✓ Models exported and tested
  ✓ Inference time acceptable (<50ms)

Code:
  ✓ Error handling implemented
  ✓ Input validation in place
  ✓ Logging configured
  ✓ Unit tests passing

Deployment:
  ✓ Infrastructure prepared
  ✓ Security measures in place
  ✓ Monitoring setup
  ✓ Backup/recovery plan
  ✓ Team trained

Documentation:
  ✓ API documented
  ✓ Model card created
  ✓ Troubleshooting guide
  ✓ Operations manual
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 14. DEBUGGING & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log predictions
result = dt.predict(5, 48.5, 25, 95, 0.051)

logger.info(f"Prediction made: SOC={result['SOC']:.3f}, Status={result['Status']}")

if result['Status'] != 'NORMAL':
    logger.warning(f"Abnormal status detected: {result['Status']}")

# Log errors
try:
    result = dt.predict(-1000, 100, 200, 500, 1.0)
except Exception as e:
    logger.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. UNIT TESTING EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

import unittest
from bms_inference import LightweightDigitalTwin

class TestDigitalTwin(unittest.TestCase):
    
    def setUp(self):
        self.dt = LightweightDigitalTwin('models')
    
    def test_prediction_shape(self):
        result = self.dt.predict(5, 48, 25, 95, 0.051)
        self.assertIn('SOC', result)
        self.assertIn('SOH', result)
        self.assertIn('Status', result)
    
    def test_soc_range(self):
        result = self.dt.predict(5, 48, 25, 95, 0.051)
        self.assertGreaterEqual(result['SOC'], 0)
        self.assertLessEqual(result['SOC'], 1)
    
    def test_status_valid(self):
        result = self.dt.predict(5, 48, 25, 95, 0.051)
        valid_statuses = ['NORMAL', 'LOW', 'CRITICAL', 'DEGRADED', 'ANOMALY']
        self.assertIn(result['Status'], valid_statuses)

if __name__ == '__main__':
    unittest.main()


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE - KEY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
DATA LOADING:
  DataGenerator.generate_test_data() - Generate synthetic data
  DataGenerator.load_excel_data(file) - Load Excel file

PREPROCESSING:
  DataPreprocessor.preprocess_time_series() - Clean/normalize
  DataPreprocessor.create_soc_from_capacity() - Calculate SOC
  DataPreprocessor.create_soh_from_cycles() - Calculate SOH

MODEL TRAINING:
  SOCModel.train() - Train LSTM for SOC
  SOHModel.train() - Train XGBoost for SOH
  ChargingTimePredictor.train() - Train RF for charging time
  DeltaVModel.train() - Train GB for voltage residuals

PREDICTIONS:
  digital_twin.predict_step() - Single prediction
  digital_twin.predict_sequence() - Multiple predictions
  LightweightDigitalTwin.predict() - Quick prediction

VISUALIZATION:
  Visualizer.plot_soc_training() - Training history
  Visualizer.plot_predictions() - Results plot
  Visualizer.plot_model_comparison() - Model accuracy

DEPLOYMENT:
  soc_model.save() - Save model
  soc_model.load() - Load model
  ModelExporter.create_deployment_config() - Create config

API:
  POST /predict - Single prediction
  POST /batch - Batch predictions
  GET /stats - Statistics
  POST /report/{id} - Health report

BMS:
  BMSInterface.receive_measurement() - Get measurement
  BMSInterface.format_for_display() - Format output
  BMSInterface.get_command() - Get control command
"""

print("✅ Quick Reference Ready!")
