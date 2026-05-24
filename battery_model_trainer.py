"""
================================================================================
BATTERY MODEL TRAINER - CUSTOM DATA TRAINING & LOGIC EXTRACTION
================================================================================
Train models on your provided battery data and extract learned logic/patterns
- Train on custom CSV/Excel files
- Build models on demand
- Extract and visualize learned logic
- Generate human-readable pattern analysis
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
# DATA VALIDATOR & LOADER
# ================================================================================

class BatteryDataValidator:
    """Validate and load battery data from CSV/Excel files"""
    
    REQUIRED_COLUMNS = {
        'voltage': ['Voltage', 'voltage', 'V', 'Volt'],
        'current': ['Current', 'current', 'I', 'Amp'],
        'temperature': ['Temperature', 'temperature', 'Temp', 'T', '°C'],
        'capacity': ['Capacity', 'capacity', 'Cap', 'Ah'],
        'resistance': ['Resistance', 'resistance', 'R', 'Ohm']
    }
    
    OPTIONAL_COLUMNS = {
        'soc': ['SOC', 'soc', 'State_of_Charge', 'state_of_charge'],
        'soh': ['SOH', 'soh', 'State_of_Health', 'state_of_health'],
        'cycle': ['Cycle', 'cycle', 'Cycle_Count', 'cycle_count'],
        'energy': ['Energy', 'energy', 'E', 'Wh'],
        'power': ['Power', 'power', 'P', 'W'],
        'timestamp': ['Timestamp', 'timestamp', 'Time', 'time', 'Date']
    }
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📂 Loading data from: {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("File must be CSV or Excel (.xlsx, .xls)")
            
            print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def validate_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Validate that required columns exist
        Returns mapping of standard names to actual column names
        """
        
        column_mapping = {}
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        # Check required columns
        for standard_name, possible_names in BatteryDataValidator.REQUIRED_COLUMNS.items():
            found = False
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    column_mapping[standard_name] = df_columns_lower[possible_name.lower()]
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Missing required column: {standard_name}")
        
        # Check optional columns
        for standard_name, possible_names in BatteryDataValidator.OPTIONAL_COLUMNS.items():
            for possible_name in possible_names:
                if possible_name.lower() in df_columns_lower:
                    column_mapping[standard_name] = df_columns_lower[possible_name.lower()]
                    break
        
        print(f"✅ Column validation successful")
        print(f"   Mapping: {column_mapping}\n")
        
        return column_mapping
    
    @staticmethod
    def clean_data(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Clean and prepare data"""
        
        print("🧹 Cleaning data...")
        
        # Remove NaN values
        initial_rows = len(df)
        df = df.dropna()
        removed_nan = initial_rows - len(df)
        
        if removed_nan > 0:
            print(f"   Removed {removed_nan} rows with NaN values")
        
        # Remove outliers (IQR method) for numeric columns
        numeric_cols = ['voltage', 'current', 'temperature', 'capacity', 'resistance']
        outliers_removed = 0
        
        for col_key in numeric_cols:
            if col_key in column_mapping:
                col = column_mapping[col_key]
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                before = len(df)
                df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
                outliers_removed += before - len(df)
        
        if outliers_removed > 0:
            print(f"   Removed {outliers_removed} outliers")
        
        print(f"✅ Cleaning complete. Final dataset: {len(df)} records\n")
        
        return df


# ================================================================================
# BATTERY MODEL TRAINER
# ================================================================================

class BatteryModelTrainer:
    """Train battery models on custom data and extract learned logic"""
    
    def __init__(self, output_dir: str = 'custom_models'):
        self.output_dir = output_dir
        self.models = {}
        self.scalers = {}
        self.logic = {}
        self.feature_importance = {}
        self.training_history = {}
        self.performance_metrics = {}
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features from raw data
        Returns: (features_df, feature_names)
        """
        
        print("🔧 Preparing features...")
        
        features_df = pd.DataFrame()
        feature_names = []
        
        # Add basic features
        basic_features = ['voltage', 'current', 'temperature', 'capacity', 'resistance']
        for feature in basic_features:
            if feature in column_mapping:
                col_name = column_mapping[feature]
                features_df[feature] = df[col_name]
                feature_names.append(feature)
        
        # Add derived features
        if 'voltage' in column_mapping and 'current' in column_mapping:
            V_col = column_mapping['voltage']
            I_col = column_mapping['current']
            
            # Power = Voltage × Current
            features_df['power'] = df[V_col] * df[I_col]
            feature_names.append('power')
            
            # Impedance (simplified)
            features_df['impedance'] = np.where(
                df[I_col] != 0,
                df[V_col] / df[I_col],
                np.nan
            ).fillna(features_df['impedance'].mean() if 'impedance' in features_df.columns else 0.05)
            feature_names.append('impedance')
        
        # Add temperature effects
        if 'temperature' in column_mapping:
            T_col = column_mapping['temperature']
            features_df['temp_squared'] = df[T_col] ** 2
            features_df['temp_cubed'] = df[T_col] ** 3
            feature_names.extend(['temp_squared', 'temp_cubed'])
        
        print(f"✅ Created {len(feature_names)} features:")
        print(f"   {feature_names}\n")
        
        return features_df, feature_names
    
    def create_targets(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Create target variables for prediction"""
        
        print("🎯 Creating target variables...")
        
        targets = {}
        
        # SOC target (if not in data, estimate from capacity)
        if 'soc' in column_mapping:
            targets['soc'] = df[column_mapping['soc']].values / 100  # Normalize to 0-1
            print("   ✓ SOC target found in data")
        else:
            if 'capacity' in column_mapping:
                cap_col = column_mapping['capacity']
                max_cap = df[cap_col].max()
                targets['soc'] = df[cap_col].values / max_cap
                print("   ✓ SOC estimated from capacity")
        
        # SOH target
        if 'soh' in column_mapping:
            targets['soh'] = df[column_mapping['soh']].values / 100  # Normalize to 0-1
            print("   ✓ SOH target found in data")
        else:
            # Estimate SOH from voltage degradation
            if 'voltage' in column_mapping:
                V_col = column_mapping['voltage']
                V_min = df[V_col].min()
                V_max = df[V_col].max()
                targets['soh'] = (df[V_col] - V_min) / (V_max - V_min)
                print("   ✓ SOH estimated from voltage")
        
        print(f"✅ Created {len(targets)} target variables\n")
        
        return targets
    
    def train_models(self, X: pd.DataFrame, targets: Dict[str, np.ndarray],
                    feature_names: List[str], test_size: float = 0.2):
        """Train multiple models and extract logic"""
        
        print("🧠 Training models...\n")
        
        # Split data
        X_train, X_test, = train_test_split(X, test_size=test_size, random_state=42)
        
        for target_name, y in targets.items():
            y_train = y[:len(X_train)]
            y_test = y[len(X_train):]
            
            print(f"📊 Training {target_name.upper()} model...")
            
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target_name] = scaler
            
            # Train Random Forest
            print(f"   Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Train Gradient Boosting
            print(f"   Training Gradient Boosting...")
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
            gb_model.fit(X_train_scaled, y_train)
            
            # Train XGBoost
            print(f"   Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            # Select best model based on R² score
            rf_score = r2_score(y_test, rf_model.predict(X_test_scaled))
            gb_score = r2_score(y_test, gb_model.predict(X_test_scaled))
            xgb_score = r2_score(y_test, xgb_model.predict(X_test_scaled))
            
            scores = {
                'Random Forest': (rf_model, rf_score),
                'Gradient Boosting': (gb_model, gb_score),
                'XGBoost': (xgb_model, xgb_score)
            }
            
            best_model_name = max(scores, key=lambda x: scores[x][1])
            best_model = scores[best_model_name][0]
            best_score = scores[best_model_name][1]
            
            self.models[target_name] = {
                'model': best_model,
                'type': best_model_name,
                'scaler': scaler
            }
            
            # Calculate metrics
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            self.performance_metrics[target_name] = {
                'model_type': best_model_name,
                'r2_score': best_score,
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            print(f"   ✓ Best Model: {best_model_name}")
            print(f"   ✓ R² Score: {best_score:.4f}")
            print(f"   ✓ RMSE: {rmse:.6f}")
            print(f"   ✓ MAE: {mae:.6f}\n")
            
            # Extract feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
                self.feature_importance[target_name] = {
                    feature_names[i]: float(importance[i])
                    for i in range(len(feature_names))
                }
    
    def extract_logic(self, feature_names: List[str]):
        """
        Extract and articulate learned patterns/logic from models
        This generates human-readable explanations of what the model learned
        """
        
        print("\n" + "="*80)
        print("🧠 EXTRACTING LEARNED LOGIC FROM MODELS")
        print("="*80 + "\n")
        
        for target_name, model_info in self.models.items():
            print(f"📍 {target_name.upper()} Model Logic:")
            print("-" * 60)
            
            model = model_info['model']
            logic_text = ""
            
            # Extract feature importance insights
            if target_name in self.feature_importance:
                importance_dict = self.feature_importance[target_name]
                
                # Sort by importance
                sorted_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                logic_text += f"\n1️⃣ FEATURE IMPORTANCE (What matters most for {target_name}):\n"
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    percentage = importance * 100
                    bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
                    logic_text += f"   {i}. {feature:20} [{bar}] {percentage:5.1f}%\n"
                
                # Key insights
                top_features = [f[0] for f in sorted_features[:3]]
                logic_text += f"\n   💡 KEY DRIVERS: {', '.join(top_features)}\n"
            
            # Extract decision rules based on feature correlations
            logic_text += f"\n2️⃣ LEARNED PATTERNS:\n"
            
            if 'voltage' in importance_dict and 'current' in importance_dict:
                if importance_dict['voltage'] > importance_dict['current']:
                    logic_text += f"   • VOLTAGE is more influential than CURRENT\n"
                    logic_text += f"     → Focus on voltage management\n"
                else:
                    logic_text += f"   • CURRENT is more influential than VOLTAGE\n"
                    logic_text += f"     → Current draw is the primary factor\n"
            
            if 'temperature' in importance_dict:
                if importance_dict['temperature'] > 0.15:
                    logic_text += f"   • TEMPERATURE is a significant factor\n"
                    logic_text += f"     → Environmental conditions matter significantly\n"
                else:
                    logic_text += f"   • TEMPERATURE has moderate influence\n"
                    logic_text += f"     → Temperature effects are secondary\n"
            
            if 'resistance' in importance_dict:
                if importance_dict['resistance'] > 0.15:
                    logic_text += f"   • INTERNAL RESISTANCE is critical\n"
                    logic_text += f"     → Battery aging/degradation is a key factor\n"
            
            # Model performance interpretation
            metrics = self.performance_metrics.get(target_name, {})
            logic_text += f"\n3️⃣ MODEL CONFIDENCE:\n"
            logic_text += f"   R² Score: {metrics.get('r2_score', 0):.4f}\n"
            logic_text += f"   RMSE: {metrics.get('rmse', 0):.6f}\n"
            
            r2 = metrics.get('r2_score', 0)
            if r2 > 0.95:
                logic_text += f"   ✅ EXCELLENT - Model explains >95% of variance\n"
            elif r2 > 0.85:
                logic_text += f"   ✅ GOOD - Model explains >85% of variance\n"
            elif r2 > 0.70:
                logic_text += f"   ⚠️  FAIR - Model explains ~70% of variance\n"
            else:
                logic_text += f"   ❌ POOR - Model needs improvement\n"
            
            # Recommendations
            logic_text += f"\n4️⃣ RECOMMENDATIONS:\n"
            if r2 < 0.80:
                logic_text += f"   • Collect more data for better training\n"
                logic_text += f"   • Review data quality and outliers\n"
                logic_text += f"   • Consider non-linear relationships\n"
            else:
                logic_text += f"   • Model is performing well\n"
                logic_text += f"   • Ready for production use\n"
                logic_text += f"   • Monitor performance over time\n"
            
            self.logic[target_name] = logic_text
            print(logic_text)
        
        return self.logic
    
    def save_logic(self, file_path: Optional[str] = None):
        """Save extracted logic to file"""
        
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'LEARNED_LOGIC.txt')
        
        with open(file_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BATTERY MODEL - LEARNED LOGIC & PATTERNS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for target_name, logic_text in self.logic.items():
                f.write(f"\n{target_name.upper()} MODEL\n")
                f.write("-" * 80 + "\n")
                f.write(logic_text)
                f.write("\n")
            
            # Save performance metrics
            f.write("\n" + "="*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            
            for target_name, metrics in self.performance_metrics.items():
                f.write(f"\n{target_name.upper()}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"\n✅ Logic saved to: {file_path}")
        return file_path
    
    def visualize_logic(self):
        """Create visualizations of learned logic"""
        
        print("\n📊 Creating visualizations...")
        
        # Feature importance plots
        for target_name, importance_dict in self.feature_importance.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            features = [features[i] for i in sorted_indices]
            importances = [importances[i] for i in sorted_indices]
            
            bars = ax.barh(features, importances, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title(f'{target_name.upper()} - Feature Importance (Learned Patterns)')
            ax.invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                       f' {importances[i]:.3f}', va='center')
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'plots', f'{target_name}_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ Saved: {plot_path}")
    
    def save_models(self):
        """Save trained models"""
        
        for target_name, model_info in self.models.items():
            model_path = os.path.join(self.output_dir, f'{target_name}_model.pkl')
            joblib.dump(model_info['model'], model_path)
            
            scaler_path = os.path.join(self.output_dir, f'{target_name}_scaler.pkl')
            joblib.dump(model_info['scaler'], scaler_path)
            
            print(f"✅ Saved {target_name} model to: {model_path}")
    
    def save_metadata(self, feature_names: List[str]):
        """Save model metadata for later use"""
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'feature_names': feature_names,
            'models': {
                name: {
                    'type': info['type'],
                    'metrics': self.performance_metrics.get(name, {})
                }
                for name, info in self.models.items()
            },
            'feature_importance': self.feature_importance,
            'logic_summary': {
                name: logic[:200] + "..." if len(logic) > 200 else logic
                for name, logic in self.logic.items()
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to: {metadata_path}")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution flow"""
    
    print("\n" + "="*80)
    print("🔋 BATTERY MODEL TRAINER - CUSTOM DATA TRAINING")
    print("="*80 + "\n")
    
    # Step 1: Load data
    print("STEP 1: LOAD YOUR DATA")
    print("-" * 80)
    data_file = input("Enter path to your battery data file (CSV or Excel): ").strip()
    
    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        return
    
    try:
        df = BatteryDataValidator.load_data(data_file)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Step 2: Validate columns
    print("\nSTEP 2: VALIDATE DATA COLUMNS")
    print("-" * 80)
    try:
        column_mapping = BatteryDataValidator.validate_columns(df)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Step 3: Clean data
    print("STEP 3: CLEAN DATA")
    print("-" * 80)
    df = BatteryDataValidator.clean_data(df, column_mapping)
    
    # Step 4: Initialize trainer
    print("\nSTEP 4: INITIALIZE MODEL TRAINER")
    print("-" * 80)
    trainer = BatteryModelTrainer(output_dir='trained_battery_models')
    
    # Step 5: Prepare features
    print("\nSTEP 5: PREPARE FEATURES")
    print("-" * 80)
    X, feature_names = trainer.prepare_features(df, column_mapping)
    
    # Step 6: Create targets
    print("\nSTEP 6: CREATE TARGETS")
    print("-" * 80)
    targets = trainer.create_targets(df, column_mapping)
    
    # Step 7: Train models
    print("\nSTEP 7: TRAIN MODELS")
    print("-" * 80)
    trainer.train_models(X, targets, feature_names)
    
    # Step 8: Extract logic
    logic = trainer.extract_logic(feature_names)
    
    # Step 9: Visualize
    print("\nSTEP 8: VISUALIZE LEARNED LOGIC")
    print("-" * 80)
    trainer.visualize_logic()
    
    # Step 10: Save everything
    print("\nSTEP 9: SAVE MODELS & LOGIC")
    print("-" * 80)
    trainer.save_models()
    trainer.save_metadata(feature_names)
    logic_file = trainer.save_logic()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📍 Output Directory: trained_battery_models/")
    print(f"   ✓ Trained models (.pkl files)")
    print(f"   ✓ Learned logic (LEARNED_LOGIC.txt)")
    print(f"   ✓ Visualizations (plots/)")
    print(f"   ✓ Metadata (model_metadata.json)")
    print(f"\n🧠 To view learned logic: cat {logic_file}")


if __name__ == "__main__":
    main()
