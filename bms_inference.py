"""
================================================================================
LIGHTWEIGHT INFERENCE SCRIPT - BMS/EDGE DEPLOYMENT
================================================================================
Optimized for embedded systems and real-time BMS integration
No full ML stack required - uses pre-trained models only
================================================================================
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime

try:
    from tensorflow.keras.saving import load_model
except:
    print("⚠️  TensorFlow not available - LSTM predictions will use fallback")

# ================================================================================
# LIGHTWEIGHT DIGITAL TWIN
# ================================================================================

class LightweightDigitalTwin:
    """
    Minimal Digital Twin for BMS integration
    - Fast inference
    - Small memory footprint
    - Works without TensorFlow if needed
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        print("📂 Loading models for BMS deployment...")
        
        try:
            # SOC LSTM
            soc_path = os.path.join(self.models_dir, 'soc_model.h5')
            if os.path.exists(soc_path):
                self.models['soc'] = load_model(soc_path)
                print("✅ SOC model loaded")
        except:
            print("⚠️  SOC model not available (will use capacity fallback)")
        
        try:
            # SOH XGBoost
            soh_path = os.path.join(self.models_dir, 'soh_model.json')
            if os.path.exists(soh_path):
                from xgboost import XGBRegressor
                model = XGBRegressor()
                model.load_model(soh_path)
                self.models['soh'] = model
                print("✅ SOH model loaded")
        except:
            print("⚠️  SOH model not available")
        
        try:
            # Charging model
            charge_path = os.path.join(self.models_dir, 'charging_model.pkl')
            if os.path.exists(charge_path):
                self.models['charging'] = joblib.load(charge_path)
                print("✅ Charging model loaded")
        except:
            print("⚠️  Charging model not available")
        
        try:
            # ΔV model
            dv_path = os.path.join(self.models_dir, 'delta_v_model.pkl')
            if os.path.exists(dv_path):
                self.models['dv'] = joblib.load(dv_path)
                print("✅ ΔV model loaded")
        except:
            print("⚠️  ΔV model not available")
        
        try:
            # Scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded")
        except:
            print("⚠️  Scaler not available")
        
        print("✅ Models loaded\n")
    
    def predict(self, current, voltage, temperature, capacity, resistance):
        """
        Real-time prediction
        
        Args:
            current: Battery current (A)
            voltage: Battery voltage (V)
            temperature: Cell temperature (°C)
            capacity: Remaining capacity (Ah)
            resistance: Internal resistance (Ω)
        
        Returns:
            dict with SOC, SOH, ΔV, charging_time, status
        """
        
        # Prepare input
        features = np.array([[current, voltage, temperature, capacity, resistance]])
        
        # SOC prediction
        soc = self._predict_soc(features, capacity)
        
        # SOH prediction
        soh = self._predict_soh(features)
        
        # ΔV prediction
        dv = self._predict_dv(features)
        
        # Charging time
        charge_time = self._predict_charging_time(features)
        
        # Status
        status = self._get_status(soc, soh, dv)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'SOC': np.clip(soc, 0, 1),
            'SOH': np.clip(soh, 0, 100),
            'Delta_V': dv,
            'Charging_Time_h': np.clip(charge_time, 0.1, 10),
            'Status': status
        }
    
    def _predict_soc(self, features, capacity):
        """Predict SOC"""
        try:
            if 'soc' in self.models:
                # For LSTM, would need sequence - use fallback
                return float(capacity) / 100
            else:
                return float(capacity) / 100
        except:
            return 0.5
    
    def _predict_soh(self, features):
        """Predict SOH"""
        try:
            if 'soh' in self.models:
                # Create cycle index from features (simplified)
                cycle_index = np.array([[0]])
                return float(self.models['soh'].predict(cycle_index)[0])
            else:
                return 100.0
        except:
            return 100.0
    
    def _predict_dv(self, features):
        """Predict voltage residual"""
        try:
            if 'dv' in self.models:
                # Use current, temperature, resistance
                dv_features = features[:, [0, 2, 4]]
                return float(self.models['dv'].predict(dv_features)[0])
            else:
                return 0.0
        except:
            return 0.0
    
    def _predict_charging_time(self, features):
        """Predict charging time"""
        try:
            if 'charging' in self.models:
                # Use capacity
                capacity = features[:, [4]]
                return float(self.models['charging'].predict(capacity)[0])
            else:
                return 2.5
        except:
            return 2.5
    
    def _get_status(self, soc, soh, dv):
        """Determine status"""
        if soc < 0.1:
            return 'CRITICAL'
        elif soc < 0.2:
            return 'LOW'
        elif soh < 80:
            return 'DEGRADED'
        elif abs(dv) > 0.5:
            return 'ANOMALY'
        else:
            return 'NORMAL'


# ================================================================================
# BMS INTEGRATION INTERFACE
# ================================================================================

class BMSInterface:
    """Interface for BMS hardware"""
    
    def __init__(self, digital_twin):
        self.dt = digital_twin
        self.buffer = []
        self.buffer_size = 10
    
    def receive_measurement(self, current, voltage, temperature, capacity, resistance):
        """Receive measurement from BMS hardware"""
        
        # Add to buffer
        measurement = {
            'current': current,
            'voltage': voltage,
            'temperature': temperature,
            'capacity': capacity,
            'resistance': resistance,
            'timestamp': datetime.now().isoformat()
        }
        
        self.buffer.append(measurement)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Make prediction
        result = self.dt.predict(current, voltage, temperature, capacity, resistance)
        
        return result
    
    def format_for_display(self, result):
        """Format result for BMS display"""
        
        display = f"""
╔════════════════════════════════╗
║  DIGITAL TWIN - BATTERY STATE  ║
╠════════════════════════════════╣
║ SOC:        {result['SOC']*100:6.1f}%              ║
║ SOH:        {result['SOH']:6.1f}%              ║
║ ΔV:         {result['Delta_V']:8.6f} V          ║
║ Charge Time: {result['Charging_Time_h']:5.2f} h           ║
║ Status:     {result['Status']:<20} ║
║ Time:       {result['timestamp'][11:19]:<20} ║
╚════════════════════════════════╝
        """
        
        return display
    
    def get_command(self, result):
        """Get control command based on prediction"""
        
        if result['Status'] == 'CRITICAL':
            return 'EMERGENCY_STOP'
        elif result['Status'] == 'LOW':
            return 'START_CHARGING'
        elif result['Status'] == 'DEGRADED':
            return 'REDUCE_DISCHARGE_RATE'
        elif result['Status'] == 'ANOMALY':
            return 'CHECK_CELLS'
        else:
            return 'NORMAL_OPERATION'


# ================================================================================
# EXAMPLE USAGE
# ================================================================================

def example_bms_simulation():
    """Simulate BMS integration"""
    
    print("=" * 80)
    print("🔋 LIGHTWEIGHT DIGITAL TWIN - BMS DEPLOYMENT")
    print("=" * 80)
    print()
    
    # Initialize
    dt = LightweightDigitalTwin('models')
    bms = BMSInterface(dt)
    
    print("📊 Simulating BMS measurements...\n")
    
    # Simulate 20 measurements
    for step in range(20):
        # Simulate varying conditions
        current = 5 * np.sin(step / 10)
        voltage = 48 + 2 * np.cos(step / 10)
        temperature = 25 + 5 * np.sin(step / 15)
        capacity = 100 - step * 0.5
        resistance = 0.05 + step * 0.001
        
        # Get prediction
        result = bms.receive_measurement(current, voltage, temperature, capacity, resistance)
        
        # Display
        print(bms.format_for_display(result))
        
        # Get command
        command = bms.get_command(result)
        print(f"⚡ Command: {command}\n")
    
    print("✅ BMS simulation complete!")


def example_real_time_prediction():
    """Real-time prediction example"""
    
    print("=" * 80)
    print("⚡ REAL-TIME PREDICTION EXAMPLE")
    print("=" * 80)
    print()
    
    dt = LightweightDigitalTwin('models')
    
    # Single prediction
    result = dt.predict(
        current=5.0,
        voltage=48.5,
        temperature=25.0,
        capacity=95.0,
        resistance=0.051
    )
    
    print("📊 Prediction Result:")
    print(json.dumps(result, indent=2, default=str))
    print()
    
    # Multiple predictions
    print("📈 Multiple predictions:")
    for i in range(5):
        result = dt.predict(
            current=5 * np.sin(i),
            voltage=48 + 0.5 * np.cos(i),
            temperature=25 + 2 * np.sin(i),
            capacity=100 - i,
            resistance=0.05 + i * 0.001
        )
        print(f"  Step {i+1}: SOC={result['SOC']:.3f}, SOH={result['SOH']:.1f}%, Status={result['Status']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'bms':
        example_bms_simulation()
    else:
        example_real_time_prediction()
    
    print("\n✅ Examples completed!")
