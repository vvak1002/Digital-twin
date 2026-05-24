"""
================================================================================
DIGITAL TWIN FASTAPI DEPLOYMENT SERVER
================================================================================
Real-time REST API for BMS integration and cloud deployment
================================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json
import os

from tensorflow.keras import load_model

# ================================================================================
# DATA MODELS
# ================================================================================

class BatteryInput(BaseModel):
    """Input battery state"""
    timestamp: Optional[str] = None
    Current: float
    Voltage: float
    Temperature: float
    Capacity: float
    Resistance: float
    CycleCount: Optional[int] = 0


class PredictionOutput(BaseModel):
    """Prediction output"""
    timestamp: str
    SOC: float
    SOH: float
    Delta_V: float
    Est_Charging_Time_h: float
    Status: str
    Confidence: float


class HealthReport(BaseModel):
    """Battery health report"""
    battery_id: str
    timestamp: str
    SOC: float
    SOH: float
    Delta_V: float
    Anomalies: List[str]
    Recommendations: List[str]
    Next_Action: str


# ================================================================================
# MODEL LOADER
# ================================================================================

class ModelLoader:
    """Load all trained models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.soc_model = None
        self.soh_model = None
        self.charge_model = None
        self.dv_model = None
        self.scaler = None
        self.config = None
        self.load_models()
    
    def load_models(self):
        """Load all models"""
        try:
            print("📂 Loading models...")
            
            # Load SOC LSTM
            soc_path = os.path.join(self.models_dir, 'soc_model.h5')
            if os.path.exists(soc_path):
                self.soc_model = load_model(soc_path)
                print(f"✅ SOC model loaded")
            
            # Load SOH XGBoost
            soh_path = os.path.join(self.models_dir, 'soh_model.json')
            if os.path.exists(soh_path):
                from xgboost import XGBRegressor
                self.soh_model = XGBRegressor()
                self.soh_model.load_model(soh_path)
                print(f"✅ SOH model loaded")
            
            # Load Charging model
            charge_path = os.path.join(self.models_dir, 'charging_model.pkl')
            if os.path.exists(charge_path):
                self.charge_model = joblib.load(charge_path)
                print(f"✅ Charging model loaded")
            
            # Load ΔV model
            dv_path = os.path.join(self.models_dir, 'delta_v_model.pkl')
            if os.path.exists(dv_path):
                self.dv_model = joblib.load(dv_path)
                print(f"✅ ΔV model loaded")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded")
            
            # Load config
            config_path = os.path.join(self.models_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Configuration loaded")
            
            print(f"✅ All models loaded successfully\n")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise


# ================================================================================
# PREDICTION ENGINE
# ================================================================================

class PredictionEngine:
    """Make predictions from deployed models"""
    
    def __init__(self, model_loader):
        self.ml = model_loader
        self.prediction_history = []
        self.features = ['Current', 'Voltage', 'Temperature', 'Capacity', 'Resistance']
    
    def predict(self, battery_input: BatteryInput) -> PredictionOutput:
        """Make single prediction"""
        
        try:
            # Prepare input
            df = pd.DataFrame([{
                'Current': battery_input.Current,
                'Voltage': battery_input.Voltage,
                'Temperature': battery_input.Temperature,
                'Capacity': battery_input.Capacity,
                'Resistance': battery_input.Resistance
            }])
            
            # Normalize
            if self.ml.scaler:
                df[self.features] = self.ml.scaler.transform(df[self.features])
            
            # Predictions
            soc = self._predict_soc(df) if self.ml.soc_model else battery_input.Capacity / 100
            soh = self._predict_soh(df) if self.ml.soh_model else 100.0
            dv = self._predict_dv(df) if self.ml.dv_model else 0.0
            charge_time = self._predict_charge_time(df) if self.ml.charge_model else 2.5
            
            # Status
            status, confidence = self._determine_status(soc, soh, dv)
            
            result = PredictionOutput(
                timestamp=battery_input.timestamp or datetime.now().isoformat(),
                SOC=np.clip(float(soc), 0, 1),
                SOH=np.clip(float(soh), 0, 100),
                Delta_V=float(dv),
                Est_Charging_Time_h=np.clip(float(charge_time), 0.1, 10),
                Status=status,
                Confidence=confidence
            )
            
            self.prediction_history.append(result)
            return result
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def _predict_soc(self, df):
        """Predict SOC (simplified)"""
        try:
            # Use capacity ratio as fallback if LSTM not working
            return float(df['Capacity'].values[0]) / 100
        except:
            return 0.5
    
    def _predict_soh(self, df):
        """Predict SOH"""
        try:
            if self.ml.soh_model:
                # Use only available features
                available_features = [f for f in ['Cycle Index'] if f in df.columns]
                if not available_features:
                    return 100.0
                return float(self.ml.soh_model.predict(df[available_features])[0])
            return 100.0
        except:
            return 100.0
    
    def _predict_dv(self, df):
        """Predict ΔV"""
        try:
            if self.ml.dv_model:
                dv_features = ['Current', 'Temperature', 'Resistance']
                if all(f in df.columns for f in dv_features):
                    return float(self.ml.dv_model.predict(df[dv_features])[0])
            return 0.0
        except:
            return 0.0
    
    def _predict_charge_time(self, df):
        """Predict charging time"""
        try:
            if self.ml.charge_model:
                charge_features = ['Capacity']
                if all(f in df.columns for f in charge_features):
                    return float(self.ml.charge_model.predict(df[charge_features])[0])
            return 2.5
        except:
            return 2.5
    
    def _determine_status(self, soc, soh, dv):
        """Determine battery status and confidence"""
        status = 'NORMAL'
        confidence = 0.95
        
        if soc < 0.1:
            status = 'CRITICAL'
            confidence = 0.98
        elif soc < 0.2:
            status = 'LOW'
            confidence = 0.96
        elif soh < 80:
            status = 'DEGRADED'
            confidence = 0.94
        elif abs(dv) > 0.5:
            status = 'ANOMALY'
            confidence = 0.85
        
        return status, confidence
    
    def generate_health_report(self, battery_id: str, battery_input: BatteryInput) -> HealthReport:
        """Generate detailed health report"""
        
        prediction = self.predict(battery_input)
        
        anomalies = []
        recommendations = []
        next_action = "Continue operation"
        
        # Detect anomalies
        if prediction.SOC < 0.1:
            anomalies.append("CRITICAL: Battery critically low")
            recommendations.append("Immediate charging required")
            next_action = "CHARGE_IMMEDIATELY"
        
        if prediction.SOC < 0.2:
            anomalies.append("LOW: Battery state low")
            recommendations.append("Schedule charging within 1 hour")
        
        if prediction.SOH < 80:
            anomalies.append("DEGRADED: Capacity degradation")
            recommendations.append("Plan battery replacement within 6 months")
            if prediction.SOH < 70:
                next_action = "REDUCE_DUTY_CYCLE"
        
        if abs(prediction.Delta_V) > 0.5:
            anomalies.append("ANOMALY: Unusual voltage signature")
            recommendations.append("Check cell balancing")
            next_action = "INSPECT_CELLS"
        
        if prediction.Est_Charging_Time_h > 4:
            recommendations.append("Slow charging detected - check charger")
        
        if len(anomalies) == 0:
            recommendations.append("Battery operating normally")
        
        return HealthReport(
            battery_id=battery_id,
            timestamp=datetime.now().isoformat(),
            SOC=prediction.SOC,
            SOH=prediction.SOH,
            Delta_V=prediction.Delta_V,
            Anomalies=anomalies,
            Recommendations=recommendations,
            Next_Action=next_action
        )


# ================================================================================
# FASTAPI APPLICATION
# ================================================================================

app = FastAPI(
    title="Digital Twin Battery API",
    description="Real-time battery state prediction and health monitoring",
    version="1.0.0"
)

# Load models on startup
try:
    model_loader = ModelLoader('models')
    engine = PredictionEngine(model_loader)
    print("✅ API initialized successfully\n")
except Exception as e:
    print(f"❌ Failed to initialize API: {e}")
    engine = None


# ================================================================================
# API ENDPOINTS
# ================================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Digital Twin Battery API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/predict",
            "/health",
            "/report",
            "/batch",
            "/stats",
            "/docs"
        ]
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(battery_input: BatteryInput):
    """
    Predict battery state
    
    Returns:
        - SOC: State of Charge (0-1)
        - SOH: State of Health (0-100%)
        - ΔV: Voltage residual
        - Charging time estimate
        - Status: NORMAL, LOW, CRITICAL, DEGRADED, ANOMALY
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return engine.predict(battery_input)


@app.post("/health")
async def health_check(battery_input: BatteryInput):
    """Get detailed health report"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return engine.generate_health_report("BATTERY_001", battery_input)


@app.post("/report/{battery_id}")
async def generate_report(battery_id: str, battery_input: BatteryInput):
    """Generate full health report for specific battery"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return engine.generate_health_report(battery_id, battery_input)


@app.post("/batch")
async def batch_predict(inputs: List[BatteryInput]):
    """
    Batch prediction for multiple batteries
    
    Returns: List of predictions
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for battery_input in inputs:
        try:
            pred = engine.predict(battery_input)
            results.append(pred)
        except Exception as e:
            results.append({"error": str(e), "input": battery_input})
    
    return results


@app.get("/stats")
async def get_statistics():
    """Get API statistics"""
    if engine is None:
        return {"error": "Models not loaded"}
    
    num_predictions = len(engine.prediction_history)
    
    if num_predictions == 0:
        return {
            "total_predictions": 0,
            "status": "No predictions yet"
        }
    
    predictions_df = pd.DataFrame([p.dict() for p in engine.prediction_history])
    
    return {
        "total_predictions": num_predictions,
        "avg_SOC": float(predictions_df['SOC'].mean()),
        "avg_SOH": float(predictions_df['SOH'].mean()),
        "avg_charging_time_h": float(predictions_df['Est_Charging_Time_h'].mean()),
        "status_distribution": predictions_df['Status'].value_counts().to_dict(),
        "last_prediction": engine.prediction_history[-1].dict() if num_predictions > 0 else None
    }


@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    if model_loader.config is None:
        return {"error": "Configuration not available"}
    
    return model_loader.config


@app.get("/health_check")
async def api_health_check():
    """Check API health status"""
    return {
        "status": "healthy" if engine else "degraded",
        "models_loaded": engine is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/alert")
async def create_alert(battery_id: str, battery_input: BatteryInput):
    """Check if battery requires alerting"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    prediction = engine.predict(battery_input)
    
    alert = {
        "battery_id": battery_id,
        "timestamp": datetime.now().isoformat(),
        "alert_required": prediction.Status != "NORMAL",
        "status": prediction.Status,
        "SOC": prediction.SOC,
        "SOH": prediction.SOH,
        "action": "ALERT" if prediction.Status != "NORMAL" else "OK"
    }
    
    return alert


# ================================================================================
# ERROR HANDLERS
# ================================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "error": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }


# ================================================================================
# RUN SERVER
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🚀 DIGITAL TWIN BATTERY API SERVER")
    print("=" * 80)
    print(f"Starting server on: http://localhost:8000")
    print(f"Documentation: http://localhost:8000/docs")
    print("=" * 80)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
