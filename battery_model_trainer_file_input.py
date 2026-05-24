"""
================================================================================
BATTERY MODEL TRAINER - CUSTOM FILE INPUT VERSION
================================================================================
Train models on uploaded battery step/cycle CSV files
Supports multiple file formats and automatically detects columns
Predicts with high accuracy based on actual battery test data
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')

# ================================================================================
# FILE FORMAT DETECTOR & LOADER
# ================================================================================

class BatteryFileLoader:
    """Detect and load various battery data file formats"""
    
    # Column name variations for different file formats
    COLUMN_MAPPINGS = {
        # Step/Cycle data format (from your file)
        'capacity_ah': ['Capacity(Ah)', 'Capacity_Ah', 'capacity', 'Cap(Ah)', 'Capacity'],
        'energy_wh': ['Energy(Wh)', 'Energy_Wh', 'energy', 'E(Wh)', 'Energy'],
        'oneset_voltage': ['Oneset Volt.(V)', 'Oneset_Voltage', 'start_voltage', 'V_start', 'Initial_Voltage'],
        'end_voltage': ['End Voltage(V)', 'End_Voltage', 'final_voltage', 'V_end', 'Final_Voltage'],
        'step_type': ['Step Type', 'Type', 'step_type', 'Mode', 'Operation'],
        'step_time': ['Step Time', 'Duration', 'Time', 'Duration(h)', 'Time_Duration'],
        'cycle_index': ['Cycle Index', 'Cycle', 'Cycle_Number', 'Cycle_Count'],
        'step_index': ['Step Index', 'Step', 'Step_Number', 'Step_ID'],
        
        # Time series format
        'voltage': ['Voltage', 'V', 'Volt', 'voltage(V)', 'Cell_Voltage'],
        'current': ['Current', 'I', 'A', 'current(A)', 'Current_A'],
        'temperature': ['Temperature', 'Temp', 'T', 'temperature(C)', 'Temp_C'],
        'timestamp': ['Timestamp', 'Time', 'DateTime', 'Date_Time', 'oneset_date'],
    }
    
    @staticmethod
    def load_file(file_path: str) -> Tuple[pd.DataFrame, str]:
        """
        Load battery data from file and detect format
        
        Returns:
            (DataFrame, format_type)
        """
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📂 Loading file: {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("File must be CSV or Excel (.xlsx, .xls)")
            
            # Detect format
            file_format = BatteryFileLoader.detect_format(df)
            
            print(f"✅ Loaded {len(df)} records")
            print(f"📋 Format detected: {file_format}")
            print(f"📊 Columns: {list(df.columns)}\n")
            
            return df, file_format
        
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """Detect if file is step/cycle or time-series format"""
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for step/cycle format
        if any('step' in col for col in columns_lower) or any('cycle' in col for col in columns_lower):
            if 'energy' in columns_lower or 'capacity' in columns_lower:
                return 'STEP_CYCLE'
        
        # Check for time-series format
        if any('timestamp' in col or 'time' in col for col in columns_lower):
            if any('voltage' in col and 'current' in col for col in columns_lower):
                return 'TIME_SERIES'
        
        # Default
        return 'UNKNOWN'
    
    @staticmethod
    def map_columns(df: pd.DataFrame) -> Dict[str, str]:
        """Map generic column names to standard names"""
        
        df_columns_lower = {col.lower(): col for col in df.columns}
        column_mapping = {}
        
        for standard_name, possible_names in BatteryFileLoader.COLUMN_MAPPINGS.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    column_mapping[standard_name] = df_columns_lower[possible_name.lower()]
                    break
        
        print("✅ Column mapping:")
        for key, value in column_mapping.items():
            print(f"   {key:20} → {value}")
        print()
        
        return column_mapping


# ================================================================================
# STEP/CYCLE DATA PROCESSOR
# ================================================================================

class StepCycleDataProcessor:
    """Process step and cycle based battery test data"""
    
    @staticmethod
    def process_step_data(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert step/cycle data to features for training
        
        Example:
            Step data with Capacity, Energy, Voltage columns
            → Extract features: Voltage drop, Energy efficiency, Capacity fade, etc.
        """
        
        print("🔧 Processing step/cycle data...")
        
        features = pd.DataFrame()
        
        # Basic parameters from each step
        if 'capacity_ah' in column_mapping:
            features['capacity'] = df[column_mapping['capacity_ah']]
        
        if 'energy_wh' in column_mapping:
            features['energy'] = df[column_mapping['energy_wh']]
        
        if 'oneset_voltage' in column_mapping and 'end_voltage' in column_mapping:
            features['voltage_start'] = df[column_mapping['oneset_voltage']]
            features['voltage_end'] = df[column_mapping['end_voltage']]
            features['voltage_drop'] = (
                df[column_mapping['oneset_voltage']] - 
                df[column_mapping['end_voltage']]
            ).abs()
        
        # Derive additional features
        if 'capacity' in features.columns and 'energy' in features.columns:
            # Efficiency = Energy / (Capacity * Voltage)
            features['efficiency'] = (
                features['energy'] / 
                (features['capacity'] * features['voltage_start'].replace(0, 1))
            )
        
        if 'voltage_drop' in features.columns and 'energy' in features.columns:
            # Power dissipation
            features['power_loss'] = features['voltage_drop'] * features['energy']
        
        # Degradation indicators
        if 'capacity' in features.columns:
            # Capacity degradation per cycle
            initial_capacity = features['capacity'].max()
            features['capacity_fade'] = (initial_capacity - features['capacity']) / initial_capacity * 100
        
        # Step-based features
        if 'step_type' in column_mapping:
            # Encode step type
            step_types = pd.Categorical(df[column_mapping['step_type']])
            for i, step_type in enumerate(step_types.categories):
                features[f'step_type_{step_type}'] = (step_types == step_type).astype(int)
        
        # Time-based features
        if 'oneset_voltage' in column_mapping and 'step_time' in column_mapping:
            try:
                # Convert step time to hours (if in HH:MM:SS format)
                if isinstance(df[column_mapping['step_time']].iloc[0], str):
                    features['step_hours'] = pd.to_timedelta(
                        df[column_mapping['step_time']]
                    ).dt.total_seconds() / 3600
                else:
                    features['step_hours'] = df[column_mapping['step_time']]
            except:
                pass
        
        print(f"✅ Created {len(features.columns)} features")
        print(f"   Features: {list(features.columns)}\n")
        
        return features
    
    @staticmethod
    def create_targets(df: pd.DataFrame, column_mapping: Dict[str, str], 
                      features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create target variables from step data
        """
        
        print("🎯 Creating target variables...")
        
        targets = {}
        
        # Target 1: SOC (State of Charge) - from voltage
        if 'voltage_end' in features.columns:
            # Normalize voltage to SOC (0-1)
            v_min = features['voltage_end'].min()
            v_max = features['voltage_end'].max()
            targets['soc'] = (features['voltage_end'] - v_min) / (v_max - v_min)
            print("   ✓ SOC target created from end voltage")
        
        # Target 2: SOH (State of Health) - from capacity
        if 'capacity_fade' in features.columns:
            # SOH = 100 - capacity_fade
            targets['soh'] = 100 - features['capacity_fade']
            print("   ✓ SOH target created from capacity fade")
        
        # Target 3: Charging efficiency
        if 'efficiency' in features.columns:
            targets['efficiency'] = features['efficiency']
            print("   ✓ Efficiency target created")
        
        # Target 4: Delta V - from voltage drop
        if 'voltage_drop' in features.columns:
            targets['delta_v'] = features['voltage_drop']
            print("   ✓ Delta V target created from voltage drop")
        
        print(f"✅ Created {len(targets)} target variables\n")
        
        return targets


# ================================================================================
# ENHANCED BATTERY MODEL TRAINER
# ================================================================================

class EnhancedBatteryModelTrainer:
    """Train on custom uploaded files with automatic format detection"""
    
    def __init__(self, output_dir: str = 'trained_battery_models'):
        self.output_dir = output_dir
        self.models = {}
        self.scalers = {}
        self.logic = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.feature_names = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    def train_from_file(self, file_path: str) -> Dict:
        """
        Complete training pipeline from file
        
        Args:
            file_path: Path to battery CSV/Excel file
        
        Returns:
            Training report dictionary
        """
        
        print("="*80)
        print("🔋 BATTERY MODEL TRAINER - FILE INPUT MODE")
        print("="*80 + "\n")
        
        # Step 1: Load file
        print("STEP 1: LOAD FILE")
        print("-"*80)
        df, file_format = BatteryFileLoader.load_file(file_path)
        
        # Step 2: Map columns
        print("STEP 2: MAP COLUMNS")
        print("-"*80)
        column_mapping = BatteryFileLoader.map_columns(df)
        
        # Step 3: Clean data
        print("STEP 3: CLEAN DATA")
        print("-"*80)
        df = self._clean_data(df, column_mapping)
        
        # Step 4: Process based on format
        print("STEP 4: PROCESS DATA")
        print("-"*80)
        if file_format == 'STEP_CYCLE':
            features = StepCycleDataProcessor.process_step_data(df, column_mapping)
        else:
            features = self._process_time_series(df, column_mapping)
        
        self.feature_names = features.columns.tolist()
        
        # Step 5: Create targets
        print("STEP 5: CREATE TARGETS")
        print("-"*80)
        targets = StepCycleDataProcessor.create_targets(df, column_mapping, features)
        
        # Step 6: Train models
        print("STEP 6: TRAIN MODELS")
        print("-"*80)
        self.train_models(features, targets)
        
        # Step 7: Extract logic
        print("STEP 7: EXTRACT LOGIC")
        print("-"*80)
        logic = self.extract_logic()
        
        # Step 8: Save
        print("STEP 8: SAVE MODELS")
        print("-"*80)
        self.save_models()
        self.save_metadata()
        self.visualize_logic()
        logic_file = self.save_logic()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n📂 Models saved to: {self.output_dir}/")
        print(f"📖 Logic saved to: {logic_file}")
        
        # Return summary
        return {
            'status': 'success',
            'models_dir': self.output_dir,
            'logic_file': logic_file,
            'features': self.feature_names,
            'targets': list(targets.keys()),
            'file_format': file_format,
            'metrics': self.performance_metrics
        }
    
    def _clean_data(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Clean and validate data"""
        
        initial_rows = len(df)
        
        # Remove NaN
        df = df.dropna()
        removed_nan = initial_rows - len(df)
        
        if removed_nan > 0:
            print(f"   Removed {removed_nan} rows with NaN")
        
        print(f"✅ Cleaned data: {len(df)} records remaining\n")
        
        return df
    
    def _process_time_series(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Process time-series data"""
        
        print("   Processing time-series format...")
        
        features = pd.DataFrame()
        
        if 'voltage' in column_mapping:
            features['voltage'] = df[column_mapping['voltage']]
        if 'current' in column_mapping:
            features['current'] = df[column_mapping['current']]
        if 'temperature' in column_mapping:
            features['temperature'] = df[column_mapping['temperature']]
        
        # Derive features
        if 'voltage' in features.columns and 'current' in features.columns:
            features['power'] = features['voltage'] * features['current']
            features['impedance'] = features['voltage'] / features['current'].replace(0, 1)
        
        print(f"✅ Created {len(features.columns)} features\n")
        
        return features
    
    def train_models(self, X: pd.DataFrame, targets: Dict[str, np.ndarray]):
        """Train models for each target"""
        
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        for target_name, y in targets.items():
            y_train = y[:len(X_train)].values
            y_test = y[len(X_train):].values
            
            print(f"🧠 Training {target_name.upper()} model...")
            
            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target_name] = scaler
            
            # Train Random Forest
            print(f"   • Random Forest...", end='')
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            rf_score = r2_score(y_test, rf.predict(X_test_scaled))
            print(f" R²={rf_score:.4f}")
            
            # Train Gradient Boosting
            print(f"   • Gradient Boosting...", end='')
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            gb.fit(X_train_scaled, y_train)
            gb_score = r2_score(y_test, gb.predict(X_test_scaled))
            print(f" R²={gb_score:.4f}")
            
            # Train XGBoost
            print(f"   • XGBoost...", end='')
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_score = r2_score(y_test, xgb_model.predict(X_test_scaled))
            print(f" R²={xgb_score:.4f}")
            
            # Select best
            scores = {'Random Forest': (rf, rf_score), 'Gradient Boosting': (gb, gb_score), 'XGBoost': (xgb_model, xgb_score)}
            best_name = max(scores, key=lambda x: scores[x][1])
            best_model = scores[best_name][0]
            best_score = scores[best_name][1]
            
            print(f"   ✓ Best: {best_name} (R²={best_score:.4f})")
            
            # Store
            self.models[target_name] = best_model
            
            # Metrics
            y_pred = best_model.predict(X_test_scaled)
            self.performance_metrics[target_name] = {
                'model': best_name,
                'r2_score': best_score,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[target_name] = {
                    X.columns[i]: float(best_model.feature_importances_[i])
                    for i in range(len(X.columns))
                }
            
            print()
    
    def extract_logic(self) -> Dict:
        """Extract learned logic from trained models"""
        
        print("🧠 EXTRACTING LEARNED LOGIC\n")
        
        logic = {}
        
        for target_name, importance_dict in self.feature_importance.items():
            logic_text = f"\n{'='*80}\n{target_name.upper()} MODEL LOGIC\n{'='*80}\n"
            
            # Feature importance
            logic_text += "\n1️⃣ FEATURE IMPORTANCE:\n"
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(sorted_features[:10], 1):
                pct = imp * 100
                bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
                logic_text += f"   {i}. {feature:20} [{bar}] {pct:5.1f}%\n"
            
            # Top drivers
            top_3 = [f[0] for f in sorted_features[:3]]
            logic_text += f"\n   💡 KEY DRIVERS: {', '.join(top_3)}\n"
            
            # Metrics
            metrics = self.performance_metrics.get(target_name, {})
            logic_text += f"\n2️⃣ MODEL PERFORMANCE:\n"
            logic_text += f"   Model Type: {metrics.get('model', 'Unknown')}\n"
            logic_text += f"   R² Score: {metrics.get('r2_score', 0):.4f}\n"
            logic_text += f"   RMSE: {metrics.get('rmse', 0):.6f}\n"
            logic_text += f"   MAE: {metrics.get('mae', 0):.6f}\n"
            
            r2 = metrics.get('r2_score', 0)
            if r2 > 0.95:
                logic_text += f"   ✅ EXCELLENT - Model explains >95% of variance\n"
            elif r2 > 0.85:
                logic_text += f"   ✅ GOOD - Model explains >85% of variance\n"
            else:
                logic_text += f"   ⚠️ FAIR - Model explains ~{r2*100:.0f}% of variance\n"
            
            logic_text += f"\n3️⃣ RECOMMENDATIONS:\n"
            if r2 > 0.85:
                logic_text += f"   • Model is ready for predictions\n"
                logic_text += f"   • Accuracy is reliable\n"
            else:
                logic_text += f"   • Collect more data for better accuracy\n"
                logic_text += f"   • Check data quality\n"
            
            self.logic[target_name] = logic_text
            print(logic_text)
        
        return self.logic
    
    def save_logic(self, file_path: Optional[str] = None) -> str:
        """Save extracted logic to file"""
        
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'LEARNED_LOGIC.txt')
        
        with open(file_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BATTERY MODEL - LEARNED LOGIC & PATTERNS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for target_name, logic_text in self.logic.items():
                f.write(logic_text)
        
        print(f"✅ Logic saved to: {file_path}\n")
        return file_path
    
    def save_models(self):
        """Save trained models"""
        
        for target_name, model in self.models.items():
            model_path = os.path.join(self.output_dir, f'{target_name}_model.pkl')
            joblib.dump(model, model_path)
            
            scaler_path = os.path.join(self.output_dir, f'{target_name}_scaler.pkl')
            joblib.dump(self.scalers[target_name], scaler_path)
            
            print(f"✅ Saved {target_name} model")
    
    def save_metadata(self):
        """Save metadata"""
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'features': self.feature_names,
            'models': {name: self.performance_metrics.get(name, {}) for name in self.models.keys()},
            'feature_importance': self.feature_importance
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Saved metadata")
    
    def visualize_logic(self):
        """Create visualizations"""
        
        for target_name, importance_dict in self.feature_importance.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            sorted_idx = np.argsort(importances)[::-1][:10]
            features = [features[i] for i in sorted_idx]
            importances = [importances[i] for i in sorted_idx]
            
            ax.barh(features, importances, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'{target_name.upper()} - Feature Importance')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'plots', f'{target_name}_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved plot: {plot_path}")


# ================================================================================
# MAIN USAGE
# ================================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("🔋 BATTERY MODEL TRAINER - FILE INPUT MODE")
    print("="*80 + "\n")
    
    # Get file path
    file_path = input("📁 Enter battery data file path (CSV or Excel): ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Train
    trainer = EnhancedBatteryModelTrainer()
    
    try:
        result = trainer.train_from_file(file_path)
        
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        print(f"\n✅ Status: {result['status'].upper()}")
        print(f"📂 Models: {result['models_dir']}")
        print(f"📖 Logic: {result['logic_file']}")
        print(f"🎯 Features: {len(result['features'])} created")
        print(f"🎯 Targets: {', '.join(result['targets'])}")
        print(f"\n📈 METRICS:")
        for target, metrics in result['metrics'].items():
            print(f"\n   {target.upper()}:")
            print(f"      R²: {metrics.get('r2_score', 0):.4f}")
            print(f"      RMSE: {metrics.get('rmse', 0):.6f}")
            print(f"      MAE: {metrics.get('mae', 0):.6f}")
        
        print("\n✅ READY TO PREDICT!")
        print("\nNext: Use battery_prediction_engine.py to make predictions")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
