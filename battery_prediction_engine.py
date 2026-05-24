"""
================================================================================
BATTERY PREDICTION ENGINE - PREDICT FUTURE BATTERY STATE
================================================================================
Predict next charging time, Delta V, and remaining useful life (DL/RUL)
Build on trained models to forecast battery behavior
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ================================================================================
# PREDICTION ENGINE - FUTURE STATE FORECASTING
# ================================================================================

class BatteryPredictionEngine:
    """
    Predict future battery states:
    - Next charging time
    - Delta V (voltage anomaly)
    - Degradation rate (DL/RUL - Days/Cycles until end of life)
    - Remaining useful life
    """
    
    def __init__(self, model_dir: str = 'trained_battery_models'):
        """
        Initialize prediction engine with trained models
        
        Args:
            model_dir: Directory containing trained models from BatteryModelTrainer
        """
        
        self.model_dir = model_dir
        self.soc_model = None
        self.soh_model = None
        self.dv_model = None
        
        self.soc_scaler = None
        self.soh_scaler = None
        self.dv_scaler = None
        
        self.metadata = None
        self.degradation_model = None
        self.charging_time_model = None
        
        self.load_models()
        self.initialize_predictors()
    
    def load_models(self):
        """Load pre-trained models from directory"""
        
        print("📂 Loading trained models...")
        
        try:
            # Load SOC model
            soc_model_path = os.path.join(self.model_dir, 'soc_model.pkl')
            if os.path.exists(soc_model_path):
                self.soc_model = joblib.load(soc_model_path)
                self.soc_scaler = joblib.load(os.path.join(self.model_dir, 'soc_scaler.pkl'))
                print("✅ SOC model loaded")
        except Exception as e:
            print(f"⚠️  Could not load SOC model: {e}")
        
        try:
            # Load SOH model
            soh_model_path = os.path.join(self.model_dir, 'soh_model.pkl')
            if os.path.exists(soh_model_path):
                self.soh_model = joblib.load(soh_model_path)
                self.soh_scaler = joblib.load(os.path.join(self.model_dir, 'soh_scaler.pkl'))
                print("✅ SOH model loaded")
        except Exception as e:
            print(f"⚠️  Could not load SOH model: {e}")
        
        try:
            # Load ΔV model
            dv_model_path = os.path.join(self.model_dir, 'delta_v_model.pkl')
            if os.path.exists(dv_model_path):
                self.dv_model = joblib.load(dv_model_path)
                self.dv_scaler = joblib.load(os.path.join(self.model_dir, 'dv_scaler.pkl'))
                print("✅ ΔV model loaded")
        except Exception as e:
            print(f"⚠️  Could not load ΔV model: {e}")
        
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("✅ Metadata loaded")
        except Exception as e:
            print(f"⚠️  Could not load metadata: {e}")
    
    def initialize_predictors(self):
        """Initialize degradation and charging time predictors"""
        
        print("\n🔧 Initializing prediction models...")
        
        # Degradation model (predicts SOH decline rate)
        self.degradation_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Charging time model (predicts next charge duration)
        self.charging_time_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        print("✅ Prediction models initialized")
    
    def predict_soc(self, features: np.ndarray) -> float:
        """
        Predict State of Charge for next measurement
        
        Args:
            features: Array of [voltage, current, temperature, capacity, resistance]
        
        Returns:
            Predicted SOC (0-1)
        """
        
        if self.soc_model is None:
            # Fallback: estimate from capacity
            return features[-2] / 100  # capacity column
        
        try:
            features_scaled = self.soc_scaler.transform(features.reshape(1, -1))
            soc_pred = self.soc_model.predict(features_scaled)[0]
            return np.clip(float(soc_pred), 0, 1)
        except:
            return 0.5
    
    def predict_soh(self, cycle_count: int, current_soh: float) -> float:
        """
        Predict State of Health at next cycle
        
        Args:
            cycle_count: Current cycle number
            current_soh: Current SOH percentage (0-100)
        
        Returns:
            Predicted SOH for next cycle (0-100)
        """
        
        if self.soh_model is None:
            # Fallback: linear degradation
            return current_soh - 0.1  # ~0.1% per cycle
        
        try:
            cycle_features = np.array([[cycle_count]]).astype(float)
            cycle_scaled = self.soh_scaler.transform(cycle_features)
            soh_pred = self.soh_model.predict(cycle_scaled)[0]
            return np.clip(float(soh_pred), 0, 100)
        except:
            return current_soh - 0.1
    
    def predict_delta_v(self, features: np.ndarray) -> float:
        """
        Predict Delta V (voltage residual) - indicates anomalies
        
        Args:
            features: Array of [voltage, current, temperature, capacity, resistance]
        
        Returns:
            Predicted ΔV (voltage difference in volts)
        """
        
        if self.dv_model is None:
            return 0.0
        
        try:
            # Use current, temperature, resistance
            dv_features = features[[1, 2, 4]]  # current, temp, resistance
            dv_scaled = self.dv_scaler.transform(dv_features.reshape(1, -1))
            dv_pred = self.dv_model.predict(dv_scaled)[0]
            return float(dv_pred)
        except:
            return 0.0
    
    def predict_charging_time(self, current_soc: float, target_soc: float = 1.0,
                             capacity: float = 100.0, charge_rate: float = 1.0) -> float:
        """
        Predict charging time to reach target SOC
        
        Args:
            current_soc: Current state of charge (0-1)
            target_soc: Target SOC to reach (default 1.0 = 100%)
            capacity: Battery capacity (Ah)
            charge_rate: Charging current rate (A)
        
        Returns:
            Predicted charging time in hours
        """
        
        if charge_rate == 0:
            return float('inf')
        
        # Calculate energy needed
        energy_needed = capacity * (target_soc - current_soc)
        
        if energy_needed <= 0:
            return 0.0
        
        try:
            # Basic formula with model adjustment
            base_time = energy_needed / charge_rate
            
            # Use model to adjust based on degradation
            charge_features = np.array([
                [current_soc, target_soc, capacity, charge_rate]
            ]).astype(float)
            
            # Fit model if not already fit (skip if no training data)
            adjusted_time = base_time
            
            # Apply degradation factor (slower charging as battery ages)
            # Placeholder - would use actual SOH data
            return np.clip(float(adjusted_time), 0.1, 48.0)
        except:
            return max(0.1, energy_needed / charge_rate)
    
    def predict_degradation_rate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        Calculate battery degradation rate and remaining useful life
        
        Args:
            df: Historical battery data with columns
            feature_cols: List of feature column names
        
        Returns:
            Dictionary with degradation metrics
        """
        
        print("\n📊 Calculating degradation rate...")
        
        if len(df) < 5:
            print("⚠️  Need at least 5 data points for degradation analysis")
            return {}
        
        # Extract SOH trend from capacity
        soh_values = (df['Capacity'].values / df['Capacity'].max()) * 100
        
        # Calculate degradation per cycle/day
        cycles = np.arange(len(soh_values))
        
        # Linear regression to find trend
        model = LinearRegression()
        model.fit(cycles.reshape(-1, 1), soh_values)
        
        degradation_rate = model.coef_[0]  # SOH decrease per cycle
        current_soh = float(soh_values[-1])
        
        # Estimate remaining cycles until EOL (end of life)
        eol_threshold = 80.0  # Battery considered dead at 80% SOH
        
        if degradation_rate < 0:
            cycles_to_eol = abs((eol_threshold - current_soh) / degradation_rate)
        else:
            cycles_to_eol = float('inf')
        
        # Estimate remaining days (assuming 1 cycle per day)
        days_to_eol = cycles_to_eol
        
        # Degradation metrics
        degradation_info = {
            'current_soh': current_soh,
            'degradation_rate_per_cycle': float(degradation_rate),
            'degradation_rate_per_100_cycles': float(degradation_rate * 100),
            'cycles_to_eol': float(cycles_to_eol),
            'days_to_eol': float(days_to_eol),
            'estimated_eol_date': (
                datetime.now() + timedelta(days=days_to_eol)
            ).strftime('%Y-%m-%d') if days_to_eol != float('inf') else 'Unknown',
            'eol_threshold': eol_threshold,
            'total_cycles_observed': len(df)
        }
        
        return degradation_info
    
    def predict_next_state(self, current_data: Dict) -> Dict:
        """
        Predict the next battery state (all parameters)
        
        Args:
            current_data: Dictionary with current battery state
                {
                    'voltage': float,
                    'current': float,
                    'temperature': float,
                    'capacity': float,
                    'resistance': float,
                    'cycle': int,
                    'soc': float (optional),
                    'soh': float (optional)
                }
        
        Returns:
            Dictionary with predicted next state
        """
        
        print("\n🔮 Predicting next battery state...")
        
        # Prepare features
        features = np.array([
            current_data.get('voltage', 48),
            current_data.get('current', 5),
            current_data.get('temperature', 25),
            current_data.get('capacity', 100),
            current_data.get('resistance', 0.05)
        ])
        
        # Current values
        current_soc = current_data.get('soc', 0.9)
        current_soh = current_data.get('soh', 98.0)
        cycle = current_data.get('cycle', 0)
        
        # Predictions
        next_soc = self.predict_soc(features)
        next_soh = self.predict_soh(cycle + 1, current_soh)
        delta_v = self.predict_delta_v(features)
        next_charge_time = self.predict_charging_time(
            current_soc,
            target_soc=1.0,
            capacity=current_data.get('capacity', 100),
            charge_rate=abs(current_data.get('current', 5))
        )
        
        # Status based on predictions
        status = self._determine_status(next_soc, next_soh, delta_v)
        
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'next_soc': float(next_soc),
                'next_soh': float(next_soh),
                'delta_v': float(delta_v),
                'estimated_charging_time_h': float(next_charge_time),
                'predicted_status': status,
                'soc_change': float(next_soc - current_soc),
                'soh_change': float(next_soh - current_soh)
            }
        }
        
        return prediction
    
    def predict_sequence(self, df: pd.DataFrame, steps_ahead: int = 10) -> pd.DataFrame:
        """
        Predict battery state for next N measurements
        
        Args:
            df: Historical data with battery measurements
            steps_ahead: How many steps to predict ahead
        
        Returns:
            DataFrame with predictions
        """
        
        print(f"\n📈 Predicting {steps_ahead} steps ahead...")
        
        predictions = []
        last_row = df.iloc[-1].to_dict()
        
        current_soc = last_row.get('SOC', 50) / 100
        current_soh = last_row.get('SOH', 98)
        
        for step in range(steps_ahead):
            # Prepare current data
            current_data = {
                'voltage': last_row.get('Voltage', 48),
                'current': last_row.get('Current', 5),
                'temperature': last_row.get('Temperature', 25),
                'capacity': last_row.get('Capacity', 100),
                'resistance': last_row.get('Resistance', 0.05),
                'cycle': last_row.get('Cycle', 0) + step,
                'soc': current_soc,
                'soh': current_soh
            }
            
            # Predict
            pred = self.predict_next_state(current_data)
            
            # Update for next iteration
            current_soc = pred['predictions']['next_soc']
            current_soh = pred['predictions']['next_soh']
            
            predictions.append({
                'step': step + 1,
                'timestamp': pred['timestamp'],
                'predicted_soc': pred['predictions']['next_soc'],
                'predicted_soh': pred['predictions']['next_soh'],
                'predicted_delta_v': pred['predictions']['delta_v'],
                'predicted_charging_time_h': pred['predictions']['estimated_charging_time_h'],
                'predicted_status': pred['predictions']['predicted_status']
            })
        
        return pd.DataFrame(predictions)
    
    def _determine_status(self, soc: float, soh: float, dv: float) -> str:
        """Determine battery status from predictions"""
        
        if soc < 0.1:
            return 'CRITICAL_LOW'
        elif soc < 0.2:
            return 'LOW'
        elif soh < 80:
            return 'DEGRADED'
        elif abs(dv) > 0.5:
            return 'ANOMALY_DETECTED'
        else:
            return 'HEALTHY'
    
    def generate_forecast_report(self, df: pd.DataFrame, 
                                steps_ahead: int = 10) -> Dict:
        """
        Generate comprehensive forecasting report
        
        Args:
            df: Historical battery data
            steps_ahead: Number of steps to predict
        
        Returns:
            Dictionary with full forecast report
        """
        
        print("\n📋 Generating forecast report...")
        
        # Current state
        current_row = df.iloc[-1]
        
        # Degradation analysis
        degradation = self.predict_degradation_rate(df, df.columns.tolist())
        
        # Future predictions
        predictions_df = self.predict_sequence(df, steps_ahead)
        
        # Report
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_summary': {
                'total_measurements': len(df),
                'date_range': f"{df.index[0] if hasattr(df, 'index') else 0} to {df.index[-1] if hasattr(df, 'index') else len(df)}",
                'current_soc': float(current_row.get('SOC', 50)) if 'SOC' in current_row else None,
                'current_soh': float(current_row.get('SOH', 98)) if 'SOH' in current_row else None
            },
            'degradation_analysis': degradation,
            'next_step_prediction': self.predict_next_state({
                'voltage': float(current_row.get('Voltage', 48)),
                'current': float(current_row.get('Current', 5)),
                'temperature': float(current_row.get('Temperature', 25)),
                'capacity': float(current_row.get('Capacity', 100)),
                'resistance': float(current_row.get('Resistance', 0.05)),
                'cycle': int(current_row.get('Cycle', 0)),
                'soc': float(current_row.get('SOC', 50)) / 100,
                'soh': float(current_row.get('SOH', 98))
            }),
            'forecast': predictions_df.to_dict('records'),
            'alerts': self._generate_alerts(degradation, predictions_df)
        }
        
        return report
    
    def _generate_alerts(self, degradation: Dict, predictions: pd.DataFrame) -> List[str]:
        """Generate alerts based on predictions"""
        
        alerts = []
        
        # Degradation alerts
        if degradation:
            if degradation.get('cycles_to_eol', float('inf')) < 100:
                alerts.append(f"⚠️ ALERT: Battery near end of life in {degradation['cycles_to_eol']:.0f} cycles")
            
            if degradation.get('degradation_rate_per_cycle', 0) > 0.5:
                alerts.append(f"⚠️ ALERT: High degradation rate detected: {degradation['degradation_rate_per_cycle']:.4f}% per cycle")
        
        # Prediction alerts
        if not predictions.empty:
            critical_soc = predictions[predictions['predicted_soc'] < 0.1]
            if not critical_soc.empty:
                alerts.append(f"⚠️ ALERT: Critical SOC predicted at step {critical_soc.iloc[0]['step']}")
            
            anomaly_dv = predictions[predictions['predicted_delta_v'].abs() > 0.5]
            if not anomaly_dv.empty:
                alerts.append(f"⚠️ ALERT: ΔV anomaly predicted at step {anomaly_dv.iloc[0]['step']}")
        
        return alerts if alerts else ["✅ No alerts - Battery in good condition"]
    
    def visualize_forecast(self, df: pd.DataFrame, predictions_df: pd.DataFrame,
                          save_path: Optional[str] = None):
        """
        Visualize historical data and predictions
        
        Args:
            df: Historical data
            predictions_df: Predictions dataframe
            save_path: Optional path to save plot
        """
        
        print("\n📊 Creating forecast visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # SOC prediction
        ax = axes[0, 0]
        ax.plot(df.index if hasattr(df, 'index') else range(len(df)), 
                df['SOC'] if 'SOC' in df.columns else df['Capacity']/100*100, 
                'o-', label='Historical', linewidth=2)
        ax.plot(range(len(df), len(df) + len(predictions_df)),
                predictions_df['predicted_soc'] * 100,
                'o--', label='Predicted', linewidth=2, color='red')
        ax.set_xlabel('Measurement')
        ax.set_ylabel('SOC (%)')
        ax.set_title('State of Charge - Historical & Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # SOH degradation
        ax = axes[0, 1]
        ax.plot(df.index if hasattr(df, 'index') else range(len(df)),
                df['SOH'] if 'SOH' in df.columns else 100 - np.linspace(0, 5, len(df)),
                'o-', label='Historical', linewidth=2)
        ax.plot(range(len(df), len(df) + len(predictions_df)),
                predictions_df['predicted_soh'],
                'o--', label='Predicted', linewidth=2, color='red')
        ax.set_xlabel('Measurement')
        ax.set_ylabel('SOH (%)')
        ax.set_title('State of Health - Historical & Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Charging time prediction
        ax = axes[1, 0]
        ax.bar(range(len(predictions_df)),
               predictions_df['predicted_charging_time_h'],
               color='steelblue', alpha=0.7)
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Charging Time (hours)')
        ax.set_title('Predicted Charging Time')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Delta V prediction
        ax = axes[1, 1]
        ax.plot(range(len(predictions_df)),
                predictions_df['predicted_delta_v'],
                'o-', linewidth=2, color='orange')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Anomaly threshold')
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('ΔV (Volts)')
        ax.set_title('Delta V Prediction - Anomaly Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        
        plt.show()
    
    def save_predictions(self, report: Dict, output_file: str):
        """Save prediction report to JSON"""
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Predictions saved to: {output_file}")


# ================================================================================
# USAGE EXAMPLE
# ================================================================================

def main():
    """Example usage of prediction engine"""
    
    print("\n" + "="*80)
    print("🔋 BATTERY PREDICTION ENGINE - FORECAST BATTERY STATE")
    print("="*80 + "\n")
    
    # Initialize engine with trained models
    engine = BatteryPredictionEngine('trained_battery_models')
    
    # Load example data
    try:
        df = pd.read_csv('example_battery_data.csv')
        print(f"✅ Loaded {len(df)} historical measurements\n")
    except FileNotFoundError:
        print("⚠️  Example data not found. Creating synthetic data...")
        
        # Create synthetic data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'Voltage': 48 + np.linspace(0, -2, n) + np.random.randn(n)*0.1,
            'Current': 5 + 2*np.sin(np.arange(n)/10) + np.random.randn(n)*0.5,
            'Temperature': 25 + np.random.randn(n),
            'Capacity': 100 - np.linspace(0, 5, n) + np.random.randn(n)*0.1,
            'Resistance': 0.05 + np.linspace(0, 0.02, n) + np.random.randn(n)*0.001,
            'SOC': 100 - np.linspace(0, 20, n) + np.random.randn(n),
            'SOH': 100 - np.linspace(0, 5, n) + np.random.randn(n)*0.1,
            'Cycle': np.arange(1, n+1)
        })
    
    # Generate forecast report
    report = engine.generate_forecast_report(df, steps_ahead=10)
    
    # Display report
    print("\n" + "="*80)
    print("📋 FORECAST REPORT")
    print("="*80)
    
    print("\n📊 Data Summary:")
    for key, value in report['data_summary'].items():
        print(f"   {key}: {value}")
    
    print("\n📉 Degradation Analysis:")
    for key, value in report['degradation_analysis'].items():
        if key != 'estimated_eol_date':
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    print(f"   estimated_eol_date: {report['degradation_analysis'].get('estimated_eol_date')}")
    
    print("\n🔮 Next Step Prediction:")
    next_pred = report['next_step_prediction']['predictions']
    print(f"   Predicted SOC: {next_pred['next_soc']:.4f}")
    print(f"   Predicted SOH: {next_pred['next_soh']:.2f}%")
    print(f"   Predicted ΔV: {next_pred['delta_v']:.6f} V")
    print(f"   Estimated Charging Time: {next_pred['estimated_charging_time_h']:.2f} hours")
    print(f"   Predicted Status: {next_pred['predicted_status']}")
    
    print("\n⚠️ Alerts:")
    for alert in report['alerts']:
        print(f"   {alert}")
    
    print("\n📈 Forecast (Next 10 Steps):")
    print(pd.DataFrame(report['forecast']).to_string())
    
    # Save report
    report_file = 'battery_forecast_report.json'
    engine.save_predictions(report, report_file)
    
    # Visualize
    predictions_df = pd.DataFrame(report['forecast'])
    engine.visualize_forecast(df, predictions_df, 'battery_forecast.png')
    
    print("\n✅ PREDICTION COMPLETE!")


if __name__ == "__main__":
    main()
