from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Telecom Churn Prediction API")

# تحديد مسار المجلد الحقيقي للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, 'churn_model (1).joblib')

# تحميل الـ Pipeline الكامل
try:
    if os.path.exists(pipeline_path):
        loaded_object = joblib.load(pipeline_path)
        if isinstance(loaded_object, dict) and 'model' in loaded_object:
            pipeline = loaded_object['model']
        else:
            pipeline = loaded_object
        print("✅ Success: Loaded Full Pipeline into memory.")
        
        # استخراج الترتيب الأصلي للأعمدة من الموديل
        try:
            if hasattr(pipeline, 'feature_names_in_'):
                expected_features = list(pipeline.feature_names_in_)
            elif hasattr(pipeline, 'steps') and hasattr(pipeline.steps[0][1], 'feature_names_in_'):
                expected_features = list(pipeline.steps[0][1].feature_names_in_)
            else:
                expected_features = [
                    'arr_network_quality_score', 'arpu_segment', 'avg_recharge_amount', 'avg_revenue_per_tx',
                    'data_volume', 'engagement_score', 'freq_top_pack', 'frequence', 'frequence_rech',
                    'is_data_user', 'is_loyal', 'log_avg_recharge_amount', 'log_avg_revenue_per_tx',
                    'log_data_volume', 'log_freq_top_pack', 'log_montant', 'log_on_net', 'log_orange',
                    'log_revenue', 'log_tigo', 'montant', 'network_quality_delta', 'no_data_flag',
                    'on_net', 'orange', 'region', 'region_avg_range', 'region_avg_samples',
                    'region_coverage_index', 'region_network_quality_score', 'region_tower_count',
                    'regularity', 'revenue', 'tigo', 'top_pack'
                ]
        except Exception:
            expected_features = None
    else:
        print(f"❌ Error: Pipeline file NOT found at {pipeline_path}")
        pipeline = None
        expected_features = None
except Exception as e:
    print(f"🚨 Critical Failure: {e}")
    pipeline = None
    expected_features = None

class CustomerDataInput(BaseModel):
    montant: float
    frequence_rech: float
    revenue: float
    arpu_segment: float
    frequence: float
    data_volume: float
    on_net: float
    orange: float
    tigo: float
    regularity: float
    freq_top_pack: float
    region_tower_count: float
    region_avg_range: float
    region_avg_samples: float
    region_coverage_index: float
    region_network_quality_score: float
    arr_network_quality_score: float

@app.post("/predict")
def predict_churn(data: CustomerDataInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="ملف الموديل مش مقروء.")
        
    try:
        df = pd.DataFrame([data.dict()])
        
        # حساب ميزات الـ Feature Engineering بدقة
        df['log_montant'] = np.log1p(df['montant'])
        df['log_revenue'] = np.log1p(df['revenue'])
        df['log_data_volume'] = np.log1p(df['data_volume'])
        df['log_on_net'] = np.log1p(df['on_net'])
        df['log_orange'] = np.log1p(df['orange'])
        df['log_tigo'] = np.log1p(df['tigo'])
        df['log_freq_top_pack'] = np.log1p(df['freq_top_pack'])
        
        df['avg_recharge_amount'] = np.where(df['frequence_rech'] <= 0, 0.0, df['montant'] / df['frequence_rech'])
        df['log_avg_recharge_amount'] = np.log1p(df['avg_recharge_amount'])
        
        df['avg_revenue_per_tx'] = np.where(df['frequence'] <= 0, 0.0, df['revenue'] / df['frequence'])
        df['log_avg_revenue_per_tx'] = np.log1p(df['avg_revenue_per_tx'])
        
        df['is_data_user'] = np.where(df['data_volume'] > 0, 1.0, 0.0)
        df['no_data_flag'] = np.where(df['data_volume'] <= 0, 1.0, 0.0)
        df['is_loyal'] = np.where(df['regularity'] > 15, 1.0, 0.0)
        df['engagement_score'] = (df['frequence'] + df['frequence_rech'] + df['regularity'])
        df['network_quality_delta'] = df['region_network_quality_score'] - df['arr_network_quality_score']
        
        df['top_pack'] = 'NO_PACK'
        df['region'] = 'UNKNOWN'

        if expected_features:
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected_features]
        else:
            backup_cols = [
                'arr_network_quality_score', 'arpu_segment', 'avg_recharge_amount', 'avg_revenue_per_tx',
                'data_volume', 'engagement_score', 'freq_top_pack', 'frequence', 'frequence_rech',
                'is_data_user', 'is_loyal', 'log_avg_recharge_amount', 'log_avg_revenue_per_tx',
                'log_data_volume', 'log_freq_top_pack', 'log_montant', 'log_on_net', 'log_orange',
                'log_revenue', 'log_tigo', 'montant', 'network_quality_delta', 'no_data_flag',
                'on_net', 'orange', 'region', 'region_avg_range', 'region_avg_samples',
                'region_coverage_index', 'region_network_quality_score', 'region_tower_count',
                'regularity', 'revenue', 'tigo', 'top_pack'
            ]
            df = df[backup_cols]
        
        probability = float(pipeline.predict_proba(df)[0][1])
        
        # عتبة القرار لقطاع الاتصالات
        if probability >= 0.25:
            risk_status = "High Risk"
            recommendation = "تحذير: العميل معرض للمغادرة! نقترح تقديم عرض مخصص لحفظ العميل."
        else:
            risk_status = "Low Risk"
            recommendation = "العميل مستقر حالياً، لا توجد خطورة."
            
        return {
            "status": "success",
            "churn_probability": round(probability, 4),
            "churn_risk": risk_status,
            "action": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في الحساب الرياضي: {str(e)}")