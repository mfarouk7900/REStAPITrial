import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import imblearn  # إجباري لفك وتشغيل الـ Pipeline

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="النسخة النهائية القاطعة - 35 عموداً بداخل Pandas DataFrame بأسماء واضحة وصحيحة."
)

# حل مشكلة المسار ديناميكياً لضمان قراءة ملف الموديل في مجلد المشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model (1).joblib")

try:
    loaded_object = joblib.load(MODEL_PATH)
    if isinstance(loaded_object, dict):
        model = loaded_object.get('model') or loaded_object.get('classifier') or loaded_object.get('pipeline')
        if model is None:
            model = list(loaded_object.values())[0]
    else:
        model = loaded_object

    if model is not None and hasattr(model, 'predict'):
        print("✅ تم تحميل الموديل والـ Pipeline بنجاح وهو في أعلى درجات الجاهزية!")
    else:
        model = None
except Exception as e:
    print(f"❌ خطأ قاتل في تحميل الموديل: {e}")
    model = None

# الـ Schema للمدخلات الـ 17 الخام القادمة من الـ Streamlit
class CustomerInput(BaseModel):
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
def predict_churn(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="نموذج الذكاء الاصطناعي غير محمل على السيرفر.")
    
    try:
        data = customer.model_dump()
        
        # 1. جلب وتأمين القيم الـ 17 الخام من الـ Streamlit
        montant = float(data['montant'])
        frequence_rech = float(data['frequence_rech'])
        revenue = float(data['revenue'])
        arpu_segment = float(data['arpu_segment'])
        frequence = float(data['frequence'])
        data_volume = float(data['data_volume'])
        on_net = float(data['on_net'])
        orange = float(data['orange'])
        tigo = float(data['tigo'])
        regularity = float(data['regularity'])
        freq_top_pack = float(data['freq_top_pack'])
        region_tower_count = float(data['region_tower_count'])
        region_avg_range = float(data['region_avg_range'])
        region_avg_samples = float(data['region_avg_samples'])
        region_coverage_index = float(data['region_coverage_index'])
        region_network_quality_score = float(data['region_network_quality_score'])
        arr_network_quality_score = float(data['arr_network_quality_score'])
        
        # 2. حساب المتغيرات الـ 18 الإضافية هندسياً بدقة بالغة
        avg_recharge_amount = montant / frequence_rech if frequence_rech > 0 else 0.0
        avg_revenue_per_tx = revenue / frequence if frequence > 0 else 0.0
        is_data_user = 1.0 if data_volume > 0 else 0.0
        no_data_flag = 1.0 if data_volume == 0 else 0.0
        is_loyal = 1.0 if regularity >= 15 else 0.0
        engagement_score = (frequence * regularity) / 30.0
        network_quality_delta = arr_network_quality_score - region_network_quality_score
        
        # التحويلات اللوغاريتمية الآمنة لمنع خطأ log(0)
        log_montant = float(np.log1p(montant))
        log_revenue = float(np.log1p(revenue))
        log_on_net = float(np.log1p(on_net))
        log_orange = float(np.log1p(orange))
        log_tigo = float(np.log1p(tigo))
        log_data_volume = float(np.log1p(data_volume))
        log_freq_top_pack = float(np.log1p(freq_top_pack))
        log_avg_recharge_amount = float(np.log1p(avg_recharge_amount))
        log_avg_revenue_per_tx = float(np.log1p(avg_revenue_per_tx))
        
        # 3. بناء قاموس يحتوي على الـ 35 عمود بالتمام والكمال وبأسماء نصية صريحة
        features_dict = {
            # الـ 17 عمود الخام
            'montant': montant, 'frequence_rech': frequence_rech, 'revenue': revenue,
            'arpu_segment': arpu_segment, 'frequence': frequence, 'data_volume': data_volume,
            'on_net': on_net, 'orange': orange, 'tigo': tigo, 'regularity': regularity,
            'freq_top_pack': freq_top_pack, 'region_tower_count': region_tower_count,
            'region_avg_range': region_avg_range, 'region_avg_samples': region_avg_samples,
            'region_coverage_index': region_coverage_index, 
            'region_network_quality_score': region_network_quality_score,
            'arr_network_quality_score': arr_network_quality_score,
            
            # الـ 18 عمود المحسوبين هندسياً واللوغاريتميين
            'log_avg_revenue_per_tx': log_avg_revenue_per_tx, 'log_revenue': log_revenue,
            'is_data_user': is_data_user, 'avg_recharge_amount': avg_recharge_amount,
            'log_tigo': log_tigo, 'log_on_net': log_on_net, 'no_data_flag': no_data_flag,
            'log_avg_recharge_amount': log_avg_recharge_amount, 'log_freq_top_pack': log_freq_top_pack,
            'avg_revenue_per_tx': avg_revenue_per_tx, 'log_data_volume': log_data_volume,
            'is_loyal': is_loyal, 'log_montant': log_montant, 'engagement_score': engagement_score,
            'log_orange': log_orange, 'network_quality_delta': network_quality_delta,
            'region': 'UNKNOWN',   # المتغير النصي المتوقع في الـ preprocessor
            'top_pack': 'UNKNOWN'  # المتغير النصي الآخر المتوقع في الـ preprocessor
        }
        
        # 4. الحل النهائي القاطع: تحويل القاموس إلى Pandas DataFrame حقيقي بأسماء الأعمدة كاملة
        input_df = pd.DataFrame([features_dict])
        
        # 5. التنبؤ الآمن وحساب الاحتمالية
        prediction = int(model.predict(input_df)[0])
        
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(input_df)[0][1])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
        churn_risk = "High Risk" if prediction == 1 else "Low Risk"
        
        return {
            "status": "success",
            "prediction": {
                "churn_risk": churn_risk,
                "churn_probability": round(probability, 4),
                "suggested_action": "إرسال عرض ترويجي مخصص وتوفير باقة إنترنت إضافية فوراً" if churn_risk == "High Risk" else "العميل مستقر حالياً"
            }
        }
    except Exception as e:
        print(f"❌ خطأ داخلي في الـ Pipeline: {str(e)}")
        raise HTTPException(status_code=400, detail=f"فشل الموديل في معالجة الـ DataFrame: {str(e)}")