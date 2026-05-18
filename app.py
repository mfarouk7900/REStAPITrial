import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Predictor")
st.markdown("---")

# 1. تحميل الموديل والـ Pipeline مباشرة جوة الـ Streamlit للـ Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, 'churn_model (1).joblib')

@st.cache_resource
def load_telecom_model():
    if os.path.exists(pipeline_path):
        loaded_object = joblib.load(pipeline_path)
        if isinstance(loaded_object, dict) and 'model' in loaded_object:
            return loaded_object['model']
        return loaded_object
    return None

pipeline = load_telecom_model()

if pipeline is None:
    st.error("🚨 خطأ: لم يتم العثور على ملف الموديل churn_model (1).joblib في مستودع GitHub!")
else:
    # استخراج الترتيب الأصلي للأعمدة من الموديل المتدرب
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

    # تصميم الواجهة واستقبال البيانات
    col1, col2 = st.columns(2)

    with col1:
        st.header("💼 Business Features")
        montant = st.number_input("Montant (Recharge)", min_value=0.0, value=100.0, step=100.0)
        frequence_rech = st.number_input("Frequence Rech", min_value=0.0, value=1.0, step=1.0)
        revenue = st.number_input("Revenue", min_value=0.0, value=50.0, step=100.0)
        arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=30.0, step=100.0)
        frequence = st.number_input("Frequence", min_value=0.0, value=1.0, step=1.0)
        data_volume = st.number_input("Data Volume", min_value=0.0, value=0.0, step=500.0)
        on_net = st.number_input("On Net", min_value=0.0, value=0.0, step=10.0)
        orange = st.number_input("Orange", min_value=0.0, value=0.0, step=10.0)
        tigo = st.number_input("Tigo", min_value=0.0, value=0.0, step=10.0)
        
        regularity = st.slider("Regularity", min_value=1.0, max_value=30.0, value=1.0, step=1.0)
        freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=0.0, step=1.0)

    with col2:
        st.header("🌐 Technical & Regional Features")
        region_tower_count = st.number_input("Region Tower Count", min_value=0.0, value=5.0, step=1.0)
        region_avg_range = st.number_input("Region Avg Range", min_value=0.0, value=1200.0, step=100.0)
        region_avg_samples = st.number_input("Region Avg Samples", min_value=0.0, value=350.0, step=50.0)
        region_coverage_index = st.number_input("Region Coverage Index", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
        
        region_network_quality_score = st.number_input("Region Network Quality Score", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
        arr_network_quality_score = st.number_input("Arrondissement Network Quality Score", min_value=0.0, max_value=1.0, value=0.85, step=0.05)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        try:
            # 2. تحويل البيانات المدخلة مباشرة إلى DataFrame
            input_dict = {
                "montant": float(montant), "frequence_rech": float(frequence_rech), "revenue": float(revenue),
                "arpu_segment": float(arpu_segment), "frequence": float(frequence), "data_volume": float(data_volume),
                "on_net": float(on_net), "orange": float(orange), "tigo": float(tigo), "regularity": float(regularity),
                "freq_top_pack": float(freq_top_pack), "region_tower_count": float(region_tower_count),
                "region_avg_range": float(region_avg_range), "region_avg_samples": float(region_avg_samples),
                "region_coverage_index": float(region_coverage_index), "region_network_quality_score": float(region_network_quality_score),
                "arr_network_quality_score": float(arr_network_quality_score)
            }
            df = pd.DataFrame([input_dict])
            
            # 3. حساب الـ Feature Engineering داخلياً
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

            # 4. التوقع الحقيقي المباشر
            probability = float(pipeline.predict_proba(df)[0][1])
            
            st.metric(label="Churn Probability", value=f"{probability:.2%}")
            
            # العتبة المتوافقة مع الموديل المعاير
            if probability >= 0.25:
                st.error("⚠️ **Prediction:** High Risk")
                st.warning("💡 تحذير: العميل معرض للمغادرة! نقترح تقديم عرض مخصص لحفظ العميل.")
            else:
                st.success("✅ **Prediction:** Low Risk")
                st.info("💡 العميل مستقر حالياً، لا توجد خطورة.")
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء الحساب الرياضي: {str(e)}")