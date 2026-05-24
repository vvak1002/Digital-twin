"""
================================================================================
BATTERY MODEL TRAINER - USAGE GUIDE & EXAMPLES
================================================================================
Complete guide to training battery models on your custom data and extracting logic
"""

# ================================================================================
# EXAMPLE 1: BASIC USAGE - SIMPLE SCRIPT
# ================================================================================

"""
STEP 1: Prepare Your Data File

Your CSV/Excel file should have columns for:
  - Voltage (V, Volt, voltage, etc.)
  - Current (A, I, Amp, current, etc.)
  - Temperature (°C, Temp, T, temperature, etc.)
  - Capacity (Ah, Cap, capacity, etc.)
  - Resistance (Ω, R, Ohm, resistance, etc.)

Example CSV format:
---
Voltage,Current,Temperature,Capacity,Resistance
48.5,5.0,25.0,95.0,0.051
48.3,4.8,25.1,94.5,0.052
48.1,4.5,25.2,94.0,0.053
...
---


STEP 2: Run the Trainer

$ python battery_model_trainer.py

Then input your file path when prompted:
> Enter path to your battery data file (CSV or Excel): /path/to/your/battery_data.csv

STEP 3: Wait for Training to Complete

The trainer will:
  ✓ Load and validate your data
  ✓ Clean and preprocess
  ✓ Create features (including derived features)
  ✓ Train 3 models (Random Forest, Gradient Boosting, XGBoost)
  ✓ Select the best model
  ✓ Extract learned logic
  ✓ Create visualizations
  ✓ Save everything

STEP 4: Check Results

All output is in: trained_battery_models/
  ✓ LEARNED_LOGIC.txt - Human-readable patterns discovered
  ✓ soc_model.pkl - Trained model for SOC prediction
  ✓ soh_model.pkl - Trained model for SOH prediction
  ✓ plots/ - Visualizations of learned patterns
  ✓ model_metadata.json - Technical metadata
"""

# ================================================================================
# EXAMPLE 2: PROGRAMMATIC USAGE - CUSTOM WORKFLOW
# ================================================================================

from battery_model_trainer import BatteryDataValidator, BatteryModelTrainer
import pandas as pd

# Load your data
data_file = 'my_battery_data.csv'
df = BatteryDataValidator.load_data(data_file)

# Validate columns
column_mapping = BatteryDataValidator.validate_columns(df)
# Returns: {'voltage': 'Voltage', 'current': 'Current', ...}

# Clean data
df = BatteryDataValidator.clean_data(df, column_mapping)

# Initialize trainer
trainer = BatteryModelTrainer(output_dir='my_models')

# Prepare features
X, feature_names = trainer.prepare_features(df, column_mapping)
print(f"Features: {feature_names}")

# Create targets
targets = trainer.create_targets(df, column_mapping)
print(f"Targets: {list(targets.keys())}")

# Train models
trainer.train_models(X, targets, feature_names, test_size=0.2)

# Extract and display logic
logic = trainer.extract_logic(feature_names)

# Save everything
trainer.save_models()
trainer.save_metadata(feature_names)
trainer.visualize_logic()
logic_file = trainer.save_logic()

# Print the learned logic
with open(logic_file, 'r') as f:
    print(f.read())


# ================================================================================
# EXAMPLE 3: RECURRING TRAINING - UPDATE MODEL ON NEW DATA
# ================================================================================

import os
from datetime import datetime

class RecurringBatteryTrainer:
    """Train model periodically and track logic evolution"""
    
    def __init__(self, data_file: str, output_base: str = 'battery_models_history'):
        self.data_file = data_file
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)
    
    def train(self, iteration: int = 1):
        """Train model and save with timestamp"""
        
        print(f"\n{'='*80}")
        print(f"TRAINING ITERATION {iteration}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.output_base, f'training_{iteration}_{timestamp}')
        
        # Load and validate
        df = BatteryDataValidator.load_data(self.data_file)
        column_mapping = BatteryDataValidator.validate_columns(df)
        df = BatteryDataValidator.clean_data(df, column_mapping)
        
        # Train
        trainer = BatteryModelTrainer(output_dir=output_dir)
        X, feature_names = trainer.prepare_features(df, column_mapping)
        targets = trainer.create_targets(df, column_mapping)
        trainer.train_models(X, targets, feature_names)
        
        # Extract and save logic
        logic = trainer.extract_logic(feature_names)
        trainer.save_models()
        trainer.save_metadata(feature_names)
        trainer.visualize_logic()
        logic_file = trainer.save_logic()
        
        # Track improvements
        self._compare_iterations(iteration)
        
        return output_dir, logic_file
    
    def _compare_iterations(self, current_iteration: int):
        """Compare performance across iterations"""
        
        if current_iteration < 2:
            return
        
        print(f"\n📊 Comparing iterations...")
        
        # Load previous and current metadata
        previous_iter = current_iteration - 1
        
        # Find directories
        dirs = sorted([d for d in os.listdir(self.output_base) 
                      if d.startswith(f'training_{current_iteration}')])
        
        if len(dirs) >= 2:
            prev_meta_file = os.path.join(self.output_base, dirs[-2], 'model_metadata.json')
            curr_meta_file = os.path.join(self.output_base, dirs[-1], 'model_metadata.json')
            
            if os.path.exists(prev_meta_file) and os.path.exists(curr_meta_file):
                import json
                
                with open(prev_meta_file) as f:
                    prev_meta = json.load(f)
                with open(curr_meta_file) as f:
                    curr_meta = json.load(f)
                
                # Compare metrics
                for model_name in curr_meta.get('models', {}):
                    prev_r2 = prev_meta['models'][model_name]['metrics'].get('r2_score', 0)
                    curr_r2 = curr_meta['models'][model_name]['metrics'].get('r2_score', 0)
                    change = curr_r2 - prev_r2
                    
                    symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"\n{symbol} {model_name.upper()}:")
                    print(f"   Previous R²: {prev_r2:.4f}")
                    print(f"   Current R²: {curr_r2:.4f}")
                    print(f"   Change: {change:+.4f}")


# Usage:
recurring_trainer = RecurringBatteryTrainer('my_battery_data.csv')

# First training
recurring_trainer.train(iteration=1)

# After collecting new data, train again
recurring_trainer.train(iteration=2)

# And again when more data arrives
recurring_trainer.train(iteration=3)

# All models and logic are saved with timestamps


# ================================================================================
# EXAMPLE 4: EXTRACT LOGIC ON DEMAND
# ================================================================================

import json

class LogicExtractor:
    """Extract logic from trained models on demand"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.logic = {}
        self.load_logic()
    
    def load_logic(self):
        """Load saved logic from file"""
        
        logic_file = os.path.join(self.model_dir, 'LEARNED_LOGIC.txt')
        
        if os.path.exists(logic_file):
            with open(logic_file, 'r') as f:
                self.logic['full'] = f.read()
            
            print(f"✅ Loaded logic from: {logic_file}")
        else:
            print(f"❌ Logic file not found: {logic_file}")
    
    def get_soc_logic(self) -> str:
        """Get SOC model logic"""
        if 'full' in self.logic:
            # Extract SOC section
            lines = self.logic['full'].split('\n')
            soc_section = []
            in_soc = False
            
            for line in lines:
                if 'SOC MODEL' in line:
                    in_soc = True
                elif 'MODEL' in line and in_soc:
                    break
                elif in_soc:
                    soc_section.append(line)
            
            return '\n'.join(soc_section)
        return ""
    
    def get_soh_logic(self) -> str:
        """Get SOH model logic"""
        if 'full' in self.logic:
            lines = self.logic['full'].split('\n')
            soh_section = []
            in_soh = False
            
            for line in lines:
                if 'SOH MODEL' in line:
                    in_soh = True
                elif 'MODEL' in line and in_soh and 'SOH' not in line:
                    break
                elif in_soh:
                    soh_section.append(line)
            
            return '\n'.join(soh_section)
        return ""
    
    def print_logic(self):
        """Print all learned logic"""
        if 'full' in self.logic:
            print(self.logic['full'])
    
    def save_logic_to_file(self, output_file: str):
        """Save logic to custom location"""
        with open(output_file, 'w') as f:
            f.write(self.logic.get('full', ''))


# Usage:
extractor = LogicExtractor('trained_battery_models')

# Print all logic whenever needed
print(extractor.get_soc_logic())
print(extractor.get_soh_logic())

# Or save to specific location
extractor.save_logic_to_file('battery_logic_report.txt')


# ================================================================================
# EXAMPLE 5: MONITOR LOGIC CHANGES OVER TIME
# ================================================================================

class LogicMonitor:
    """Track how learned logic changes as model trains on new data"""
    
    def __init__(self, base_dir: str = 'battery_models_history'):
        self.base_dir = base_dir
        self.history = []
    
    def load_training_history(self):
        """Load all training iterations"""
        
        training_dirs = sorted([
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ])
        
        for train_dir in training_dirs:
            metadata_file = os.path.join(self.base_dir, train_dir, 'model_metadata.json')
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.history.append({
                    'iteration': train_dir,
                    'timestamp': metadata.get('timestamp'),
                    'feature_importance': metadata.get('feature_importance'),
                    'metrics': metadata.get('models')
                })
        
        print(f"✅ Loaded {len(self.history)} training iterations")
    
    def compare_feature_importance(self):
        """Compare feature importance across iterations"""
        
        print("\n📊 FEATURE IMPORTANCE CHANGES ACROSS ITERATIONS:\n")
        
        if not self.history:
            self.load_training_history()
        
        for i, record in enumerate(self.history, 1):
            print(f"\n{i}. Iteration: {record['iteration']}")
            
            if 'soc' in record.get('feature_importance', {}):
                print("   SOC Model Top Features:")
                features = record['feature_importance']['soc']
                
                sorted_features = sorted(
                    features.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for feat, importance in sorted_features:
                    print(f"      • {feat}: {importance:.3f}")
    
    def show_improvement_trends(self):
        """Show R² improvement trends"""
        
        print("\n📈 MODEL PERFORMANCE TRENDS:\n")
        
        if not self.history:
            self.load_training_history()
        
        for i, record in enumerate(self.history, 1):
            print(f"\n{i}. Iteration {i}:")
            
            if 'soc' in record.get('metrics', {}):
                r2 = record['metrics']['soc'].get('r2_score', 0)
                rmse = record['metrics']['soc'].get('rmse', 0)
                print(f"   SOC:  R²={r2:.4f}, RMSE={rmse:.6f}")
            
            if 'soh' in record.get('metrics', {}):
                r2 = record['metrics']['soh'].get('r2_score', 0)
                rmse = record['metrics']['soh'].get('rmse', 0)
                print(f"   SOH:  R²={r2:.4f}, RMSE={rmse:.6f}")


# Usage:
monitor = LogicMonitor('battery_models_history')
monitor.compare_feature_importance()
monitor.show_improvement_trends()


# ================================================================================
# EXAMPLE 6: API ENDPOINT TO QUERY LOGIC ON DEMAND
# ================================================================================

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Battery Logic API")

class LogicRequest(BaseModel):
    model_type: str = "soc"  # "soc" or "soh"
    detail_level: str = "full"  # "summary" or "full"

class LogicResponse(BaseModel):
    model_type: str
    logic: str
    timestamp: str

logic_cache = {}

@app.get("/logic/soc")
async def get_soc_logic():
    """Get SOC model learned logic"""
    
    extractor = LogicExtractor('trained_battery_models')
    return LogicResponse(
        model_type="soc",
        logic=extractor.get_soc_logic(),
        timestamp=datetime.now().isoformat()
    )

@app.get("/logic/soh")
async def get_soh_logic():
    """Get SOH model learned logic"""
    
    extractor = LogicExtractor('trained_battery_models')
    return LogicResponse(
        model_type="soh",
        logic=extractor.get_soh_logic(),
        timestamp=datetime.now().isoformat()
    )

@app.get("/logic/all")
async def get_all_logic():
    """Get complete learned logic"""
    
    extractor = LogicExtractor('trained_battery_models')
    return LogicResponse(
        model_type="all",
        logic=extractor.logic.get('full', ''),
        timestamp=datetime.now().isoformat()
    )

@app.post("/train")
async def trigger_training(data_file: str = 'my_battery_data.csv'):
    """Trigger model training on new data"""
    
    recurring_trainer = RecurringBatteryTrainer(data_file)
    iteration = 1  # Get current iteration count
    
    output_dir, logic_file = recurring_trainer.train(iteration=iteration)
    
    with open(logic_file, 'r') as f:
        logic = f.read()
    
    return {
        'status': 'training_complete',
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat(),
        'logic_preview': logic[:500]  # First 500 chars
    }

# Run with: uvicorn battery_logic_api:app --reload


# ================================================================================
# EXAMPLE 7: CUSTOM DATA FORMAT EXAMPLE
# ================================================================================

"""
Create a CSV file with your battery data:

Filename: battery_data.csv
---
Voltage,Current,Temperature,Capacity,Resistance,SOC,Cycle
48.5,5.0,25.0,95.0,0.051,95,100
48.3,4.8,25.1,94.5,0.052,94,100
48.1,4.5,25.2,94.0,0.053,93,100
47.9,4.2,25.3,93.5,0.054,92,100
47.7,4.0,25.4,93.0,0.055,91,100
47.5,3.8,25.5,92.5,0.056,90,100
47.3,3.5,25.6,92.0,0.057,89,100
---

Column explanations:
- Voltage: Battery output voltage in volts
- Current: Charging/discharging current in amperes
- Temperature: Battery temperature in Celsius
- Capacity: Available capacity in Ah (amp-hours)
- Resistance: Internal resistance in ohms
- SOC: State of Charge in percentage (0-100)
- Cycle: Cycle count (how many times charged/discharged)

The model will:
1. Use Voltage, Current, Temperature, Capacity, Resistance as FEATURES
2. Predict SOC and SOH (inferred from capacity/voltage trends)
3. Learn which features matter most
4. Generate patterns explaining the relationships
"""


# ================================================================================
# EXAMPLE 8: WORKFLOW SUMMARY
# ================================================================================

"""
COMPLETE WORKFLOW FOR RECURRING TRAINING + LOGIC EXTRACTION:

1. INITIAL SETUP
   ├─ Prepare battery_data.csv with columns
   └─ Run: python battery_model_trainer.py

2. FIRST TRAINING
   ├─ System trains on your data
   ├─ Creates trained_battery_models/ directory
   ├─ Generates LEARNED_LOGIC.txt with patterns found
   └─ Saves models for reuse

3. ON DEMAND LOGIC QUERY
   ├─ Use LogicExtractor class to read logic
   ├─ Query specific model (SOC/SOH)
   └─ Display patterns without retraining

4. NEW DATA ARRIVES
   ├─ Add new rows to battery_data.csv
   ├─ Use RecurringBatteryTrainer to retrain
   ├─ Compare improvements vs previous training
   └─ Updated logic reflects new patterns

5. API ENDPOINT (Optional)
   ├─ Run FastAPI server
   ├─ Query logic via HTTP: GET /logic/soc
   ├─ Trigger training: POST /train
   └─ Monitor changes: GET /logic/history

EXAMPLE DAILY SCHEDULE:
├─ 00:00 - Collect daily battery test data
├─ 01:00 - Run training: recurring_trainer.train(iteration=N)
├─ 02:00 - Extract logic: extractor.get_soc_logic()
├─ 02:30 - Generate report
└─ 03:00 - Alert if logic changed significantly


KEY BENEFITS:
✅ Trains only when you run it (not automatic)
✅ Extracts human-readable logic from trained models
✅ Tracks logic evolution over time
✅ Can query specific model patterns on demand
✅ Compares performance improvements
✅ Fully customizable - your data, your schedule
"""

print(__doc__)
