"""
================================================================================
QUICK START DEMO - DIGITAL TWIN PIPELINE
================================================================================
Simple usage examples showing how to use the complete pipeline
================================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from digital_twin_pipeline import (
    DataGenerator, DataPreprocessor,
    SOCModel, SOHModel, ChargingTimePredictor, DeltaVModel,
    DigitalTwin, Visualizer
)

# ================================================================================
# DEMO 1: QUICK START
# ================================================================================

def demo_quick_start():
    """Fastest way to get started"""
    
    print("\n" + "=" * 80)
    print("🚀 DEMO 1: QUICK START (5 minutes)")
    print("=" * 80)
    print()
    
    # 1. Generate data
    print("Step 1: Generating data...")
    df_time, df_cycle, df_step, df_test = DataGenerator.generate_test_data(
        num_samples=2000, num_cycles=30
    )
    print(f"✅ Generated {len(df_time)} time records\n")
    
    # 2. Preprocess
    print("Step 2: Preprocessing...")
    preprocessor = DataPreprocessor()
    features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
    df_time = preprocessor.preprocess_time_series(df_time, features)
    df_time = preprocessor.create_soc_from_capacity(df_time)
    df_cycle = preprocessor.preprocess_cycle_data(df_cycle)
    df_cycle = preprocessor.create_soh_from_cycles(df_cycle)
    print("✅ Preprocessing complete\n")
    
    # 3. Train models (mini)
    print("Step 3: Training models (fast mode)...")
    soc_model = SOCModel(sequence_length=20, features_dim=len(features))
    soc_model.train(df_time, features, epochs=5, batch_size=128)
    
    soh_model = SOHModel()
    soh_model.train(df_cycle)
    
    charge_model = ChargingTimePredictor()
    charge_model.train(df_step)
    
    dv_model = DeltaVModel()
    dv_model.train(df_time)
    print("✅ All models trained\n")
    
    # 4. Create Digital Twin
    print("Step 4: Creating Digital Twin...")
    digital_twin = DigitalTwin(soc_model, soh_model, charge_model, dv_model, preprocessor)
    print("✅ Digital Twin ready\n")
    
    # 5. Make predictions
    print("Step 5: Real-time predictions...")
    for i in range(1000, 1010):
        row = df_time.iloc[i]
        input_data = {
            'Current': row['Current'],
            'Voltage': row['Voltage'],
            'Temperature': row['Temperature'],
            'Capacity': row['Capacity'],
            'Resistance': row['Resistance']
        }
        result = digital_twin.predict_step(input_data)
        print(f"  {i-999}: SOC={result['SOC']:.3f}, SOH={result['SOH']:.1f}%, Status={result['Status']}")
    
    print("\n✅ DEMO 1 COMPLETE!")


# ================================================================================
# DEMO 2: CUSTOM DATA
# ================================================================================

def demo_custom_data():
    """Use custom/real data"""
    
    print("\n" + "=" * 80)
    print("📊 DEMO 2: LOAD YOUR OWN DATA")
    print("=" * 80)
    print()
    
    # Try to load Excel file
    excel_file = r"D:\OneDrive - Matter Motor Works Pvt. Ltd\M1 (LISHEN) BATTERY PACK (100V50AH) DVP\M1 LISHEN DVP TEST Raw Data\Lishen LIFE CYCLE TEST (RT)\MM01300389005G00006\923-MM0130089005G00006-BTS85-227-1-3-67.xlsx"
    
    if os.path.exists(excel_file):
        print(f"📂 Loading {excel_file}...")
        df_time, df_cycle, df_step, df_test = DataGenerator.load_excel_data(excel_file)
        print("✅ Excel data loaded\n")
    else:
        print(f"⚠️  {excel_file} not found")
        print("ℹ️  Place your Excel file in the current directory")
        print("    File should have sheets: time_series, test, cycle, step\n")
        return
    
    # Show data info
    print("📊 Data Summary:")
    print(f"  Time Series: {len(df_time)} records")
    print(f"  - Columns: {list(df_time.columns)}\n")
    print(f"  Cycles: {len(df_cycle)} records")
    print(f"  - Columns: {list(df_cycle.columns)}\n")
    print(f"  Steps: {len(df_step)} records")
    print(f"  - Columns: {list(df_step.columns)}\n")


# ================================================================================
# DEMO 3: COMPARE MODELS
# ================================================================================

def demo_model_comparison():
    """Train models on same data and compare"""
    
    print("\n" + "=" * 80)
    print("🧠 DEMO 3: MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    
    # Generate data
    print("Generating test data...")
    df_time, df_cycle, df_step, df_test = DataGenerator.generate_test_data(
        num_samples=3000, num_cycles=40
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
    df_time = preprocessor.preprocess_time_series(df_time, features)
    df_time = preprocessor.create_soc_from_capacity(df_time)
    df_cycle = preprocessor.preprocess_cycle_data(df_cycle)
    df_cycle = preprocessor.create_soh_from_cycles(df_cycle)
    
    print("Training models...\n")
    
    # Train SOC model
    print("📊 SOC Model Performance:")
    soc_model = SOCModel(sequence_length=20, features_dim=len(features))
    X_soc_val, y_soc_val = soc_model.train(df_time, features, epochs=3, batch_size=128)
    
    if len(X_soc_val) > 0:
        soc_pred = soc_model.predict(X_soc_val).flatten()
        soc_mse = np.mean((y_soc_val - soc_pred) ** 2)
        soc_mae = np.mean(np.abs(y_soc_val - soc_pred))
        print(f"  MSE: {soc_mse:.6f}")
        print(f"  MAE: {soc_mae:.6f}\n")
    
    # Train SOH model
    print("📊 SOH Model Performance:")
    soh_model = SOHModel()
    X_soh_val, y_soh_val = soh_model.train(df_cycle)
    
    if len(X_soh_val) > 0:
        soh_pred = soh_model.predict(X_soh_val)
        soh_mse = np.mean((y_soh_val - soh_pred) ** 2)
        soh_mae = np.mean(np.abs(y_soh_val - soh_pred))
        print(f"  MSE: {soh_mse:.6f}")
        print(f"  MAE: {soh_mae:.6f}\n")


# ================================================================================
# DEMO 4: REAL-TIME SIMULATION
# ================================================================================

def demo_real_time_simulation():
    """Full real-time simulation"""
    
    print("\n" + "=" * 80)
    print("⏱️ DEMO 4: REAL-TIME SIMULATION (Full Pipeline)")
    print("=" * 80)
    print()
    
    # Load/generate data
    excel_file = "923-MM0130089005G00006-BTS85-227-1-3-67.xlsx"
    if os.path.exists(excel_file):
        print(f"📂 Loading {excel_file}...")
        df_time, df_cycle, df_step, df_test = DataGenerator.load_excel_data(excel_file)
    else:
        print("📊 Generating synthetic data...")
        df_time, df_cycle, df_step, df_test = DataGenerator.generate_test_data(
            num_samples=5000, num_cycles=50
        )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
    df_time = preprocessor.preprocess_time_series(df_time, features)
    df_time = preprocessor.create_soc_from_capacity(df_time)
    df_cycle = preprocessor.preprocess_cycle_data(df_cycle)
    df_cycle = preprocessor.create_soh_from_cycles(df_cycle)
    
    # Train all models
    print("\n🧠 Training models...")
    soc_model = SOCModel(sequence_length=20, features_dim=len(features))
    soc_model.train(df_time, features, epochs=10, batch_size=64)
    
    soh_model = SOHModel()
    soh_model.train(df_cycle)
    
    charge_model = ChargingTimePredictor()
    charge_model.train(df_step)
    
    dv_model = DeltaVModel()
    dv_model.train(df_time)
    
    # Create Digital Twin
    print("\n🔧 Creating Digital Twin...")
    digital_twin = DigitalTwin(soc_model, soh_model, charge_model, dv_model, preprocessor)
    
    # Run simulation
    print("\n⏱️ Running 100 real-time predictions...")
    results_df = digital_twin.predict_sequence(df_time, features, start_idx=1000, num_steps=100)
    
    print("\nResults Summary:")
    print(results_df[['SOC', 'SOH', 'Status']].describe())
    
    # Visualize
    print("\n📊 Generating visualizations...")
    Visualizer.plot_predictions(results_df)
    
    print("\n✅ DEMO 4 COMPLETE!")


# ================================================================================
# DEMO 5: SAVE & LOAD MODELS
# ================================================================================

def demo_save_load():
    """Train, save, and load models"""
    
    print("\n" + "=" * 80)
    print("💾 DEMO 5: SAVE & LOAD MODELS")
    print("=" * 80)
    print()
    
    # Generate data
    print("Generating data...")
    df_time, df_cycle, df_step, df_test = DataGenerator.generate_test_data(
        num_samples=2000, num_cycles=25
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
    df_time = preprocessor.preprocess_time_series(df_time, features)
    df_time = preprocessor.create_soc_from_capacity(df_time)
    df_cycle = preprocessor.preprocess_cycle_data(df_cycle)
    df_cycle = preprocessor.create_soh_from_cycles(df_cycle)
    
    # Train models
    print("Training models...")
    soc_model = SOCModel(sequence_length=20, features_dim=len(features))
    soc_model.train(df_time, features, epochs=5, batch_size=128)
    
    # Save
    print("\n💾 Saving models...")
    soc_model.save('models/soc_model_demo.h5')
    
    # Load
    print("📂 Loading models...")
    soc_model_loaded = SOCModel()
    soc_model_loaded.load('models/soc_model_demo.h5')
    
    print("✅ Model saved and loaded successfully!")


# ================================================================================
# DEMO 6: QUICK PREDICTIONS
# ================================================================================

def demo_quick_predictions():
    """Make quick predictions without full training"""
    
    print("\n" + "=" * 80)
    print("⚡ DEMO 6: QUICK PREDICTIONS (No Training)")
    print("=" * 80)
    print()
    
    print("Using pre-trained models from 'models/' directory\n")
    
    # Try to load pre-trained models
    from bms_inference import LightweightDigitalTwin
    
    try:
        dt = LightweightDigitalTwin('models')
        print("✅ Models loaded\n")
    except Exception as e:
        print(f"⚠️  Could not load models: {e}")
        print("Run DEMO 1 or DEMO 4 first to generate models\n")
        return
    
    # Make predictions
    print("📊 Sample predictions:\n")
    
    test_cases = [
        {"name": "Normal", "current": 5, "voltage": 48, "temp": 25, "capacity": 95, "resistance": 0.051},
        {"name": "Charging", "current": 10, "voltage": 52, "temp": 28, "capacity": 85, "resistance": 0.052},
        {"name": "Discharging", "current": -5, "voltage": 44, "temp": 22, "capacity": 50, "resistance": 0.053},
        {"name": "Critical", "current": 2, "voltage": 40, "temp": 30, "capacity": 5, "resistance": 0.060},
    ]
    
    for case in test_cases:
        result = dt.predict(
            current=case['current'],
            voltage=case['voltage'],
            temperature=case['temp'],
            capacity=case['capacity'],
            resistance=case['resistance']
        )
        print(f"  {case['name']:12} → SOC={result['SOC']:.2%}, SOH={result['SOH']:.0f}%, Status={result['Status']}")
    
    print("\n✅ DEMO 6 COMPLETE!")


# ================================================================================
# MAIN MENU
# ================================================================================

def main():
    """Main demo menu"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    DIGITAL TWIN PIPELINE - DEMOS                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Choose a demo to run:

  1. QUICK START (5 min)
     └─ Minimal example, generates synthetic data, trains all models

  2. CUSTOM DATA
     └─ Load and analyze your Excel file

  3. MODEL COMPARISON
     └─ Compare performance of trained models

  4. REAL-TIME SIMULATION
     └─ Full pipeline with visualization

  5. SAVE & LOAD MODELS
     └─ Train, save, and load models

  6. QUICK PREDICTIONS
     └─ Use pre-trained models for instant predictions

  0. EXIT

    """)
    
    while True:
        choice = input("Enter choice (0-6): ").strip()
        
        if choice == '0':
            print("\nGoodbye! 👋\n")
            break
        elif choice == '1':
            demo_quick_start()
        elif choice == '2':
            demo_custom_data()
        elif choice == '3':
            demo_model_comparison()
        elif choice == '4':
            demo_real_time_simulation()
        elif choice == '5':
            demo_save_load()
        elif choice == '6':
            demo_quick_predictions()
        else:
            print("❌ Invalid choice. Try again.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific demo
        demo_name = sys.argv[1].lower()
        if demo_name == '1' or demo_name == 'quick':
            demo_quick_start()
        elif demo_name == '2' or demo_name == 'custom':
            demo_custom_data()
        elif demo_name == '3' or demo_name == 'compare':
            demo_model_comparison()
        elif demo_name == '4' or demo_name == 'simulation':
            demo_real_time_simulation()
        elif demo_name == '5' or demo_name == 'save':
            demo_save_load()
        elif demo_name == '6' or demo_name == 'predict':
            demo_quick_predictions()
        else:
            print(f"Unknown demo: {demo_name}")
    else:
        # Interactive menu
        main()
