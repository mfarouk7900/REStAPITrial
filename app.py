import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import imblearn  # إجباري لفك وتشغيل الـ Pipeline

st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# 🌟 خطوة تحميل الموديل مباشرة داخل الـ Streamlit دون الحاجة لسيرفر خارجي
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model (1).joblib")

@st.cache_resource # لتسريع التطبيق وتحميل الموديل مرة واحدة فقط في الذاكرة
def load_my_model():
    try:
        loaded_object = joblib.load(MODEL_PATH)
        if isinstance(loaded_object, dict):
            m = loaded_object.get('model') or loaded_object.get('classifier') or loaded_object.get('pipeline')
            if m is None:
                m = list(loaded_object.values())[0]
        else:
            m = loaded_object
        return m
    except Exception as e:
        st.error(f"❌ خطأ في تحميل ملف الموديل بالسيرفر: {e}")
        return None

model = load_my_model()

st.title("📱 AI-Enhanced Data Pipeline for Customer Churn Prediction")
st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Single Customer Prediction", "📈 Financial Feasibility & ROI"])

with tab1:
    st.header("Single Customer Risk Assessment")
    st.subheader("Enter Customer Usage & Network Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Financial & Customer Activity")
        montant = st.number_input("Montant (Recharge Amount)", value=5000.0)
        frequence_rech = st.number_input("Frequence Recharge count", value=10.0)
        revenue = st.number_input("Revenue generated", value=5500.0)
        arpu_segment = st.number_input("ARPU Segment", value=1800.0)
        frequence = st.number_input("Frequence (Activity frequency)", value=12.0)
        regularity = st.number_input("Regularity (Days active)", value=30.0)
        freq_top_pack = st.number_input("Freq Top Pack (Most used packages)", value=5.0)

    with col2:
        st.markdown("### 📶 Network Quality Metrics")
        data_volume = st.number_input("Data Volume (MB used)", value=1500.0)
        on_net = st.number_input("On Net Calls (Same network)", value=50.0)
        orange = st.number_input("Calls to Orange Network", value=25.0)
        tigo = st.number_input("Calls to Tigo Network", value=15.0)

    with col3:
        st.markdown("### 🏢 Regional Infrastructure")
        region_tower_count = st.number_input("Region Tower Count", value=14.0)
        region_avg_range = st.number_input("Region Avg Tower Range", value=2.5)
        region_avg_samples = st.number_input("Region Avg Signal Samples", value=120.0)
        region_coverage_index = st.number_input("Region Coverage Index", value=0.85)
        region_network_quality_score = st.number_input("Region Network Quality Score", value=78.0)
        arr_network_quality_score = st.number_input("Arrondissement Network Quality Score", value=82.0)

    if st.button("🔮 Predict Churn Risk"):
        if model is None:
            st.error("النموذج غير محمل بشكل صحيح على السيرفر.")
        else:
            try:
                with st.spinner("جاري حساب الـ Feature Engineering والـ 35 متغير داخلياً..."):
                    # 1. تأمين تحويل الأنواع الرقمية
                    m_val = float(montant)
                    f_rech_val = float(frequence_rech)
                    rev_val = float(revenue)
                    arpu_val = float(arpu_segment)
                    freq_val = float(frequence)
                    data_val = float(data_volume)
                    on_net_val = float(on_net)
                    orange_val = float(orange)
                    tigo_val = float(tigo)
                    reg_val = float(regularity)
                    top_pack_val = float(freq_top_pack)
                    
                    # 2. حساب المتغيرات الـ 18 الإضافية هندسياً بدقة
                    avg_recharge_amount = m_val / f_rech_val if f_rech_val > 0 else 0.0
                    avg_revenue_per_tx = rev_val / freq_val if freq_val > 0 else 0.0
                    is_data_user = 1.0 if data_val > 0 else 0.0
                    no_data_flag = 1.0 if data_val == 0 else 0.0
                    is_loyal = 1.0 if reg_val >= 15 else 0.0
                    engagement_score = (freq_val * reg_val) / 30.0
                    network_quality_delta = float(arr_network_quality_score) - float(region_network_quality_score)
                    
                    # التحويلات اللوغاريتمية الآمنة
                    log_montant = float(np.log1p(m_val))
                    log_revenue = float(np.log1p(rev_val))
                    log_on_net = float(np.log1p(on_net_val))
                    log_orange = float(np.log1p(orange_val))
                    log_tigo = float(np.log1p(tigo_val))
                    log_data_volume = float(np.log1p(data_val))
                    log_freq_top_pack = float(np.log1p(top_pack_val))
                    log_avg_recharge_amount = float(np.log1p(avg_recharge_amount))
                    log_avg_revenue_per_tx = float(np.log1p(avg_revenue_per_tx))
                    
                    # 3. بناء الـ DataFrame الكامل بالـ 35 عمود بالأسماء النصية الصارمة للموديل
                    features_dict = {
                        'montant': m_val, 'frequence_rech': f_rech_val, 'revenue': rev_val,
                        'arpu_segment': arpu_val, 'frequence': freq_val, 'data_volume': data_val,
                        'on_net': on_net_val, 'orange': orange_val, 'tigo': tigo_val, 'regularity': reg_val,
                        'freq_top_pack': top_pack_val, 'region_tower_count': float(region_tower_count),
                        'region_avg_range': float(region_avg_range), 'region_avg_samples': float(region_avg_samples),
                        'region_coverage_index': float(region_coverage_index), 
                        'region_network_quality_score': float(region_network_quality_score),
                        'arr_network_quality_score': float(arr_network_quality_score),
                        
                        'log_avg_revenue_per_tx': log_avg_revenue_per_tx, 'log_revenue': log_revenue,
                        'is_data_user': is_data_user, 'avg_recharge_amount': avg_recharge_amount,
                        'log_tigo': log_tigo, 'log_on_net': log_on_net, 'no_data_flag': no_data_flag,
                        'log_avg_recharge_amount': log_avg_recharge_amount, 'log_freq_top_pack': log_freq_top_pack,
                        'avg_revenue_per_tx': avg_revenue_per_tx, 'log_data_volume': log_data_volume,
                        'is_loyal': is_loyal, 'log_montant': log_montant, 'engagement_score': engagement_score,
                        'log_orange': log_orange, 'network_quality_delta': network_quality_delta,
                        'region': 'UNKNOWN',
                        'top_pack': 'UNKNOWN'
                    }
                    
                    input_df = pd.DataFrame([features_dict])
                    
                    # 4. التنبؤ مباشرة
                    prediction = int(model.predict(input_df)[0])
                    probability = float(model.predict_proba(input_df)[0][1]) if hasattr(model, 'predict_proba') else (1.0 if prediction == 1 else 0.0)
                    
                    churn_risk = "High Risk" if prediction == 1 else "Low Risk"
                    
                    st.markdown("---")
                    if churn_risk == "High Risk":
                        st.error(f"⚠️ **Prediction: {churn_risk}**")
                    else:
                        st.success(f"✅ **Prediction: {churn_risk}**")
                        
                    st.metric(label="Churn Probability", value=f"{round(probability * 100, 2)}%")
                    st.info(f"💡 **Suggested Corporate Action:** {'إرسال عرض ترويجي مخصص وتوفير باقة إنترنت إضافية' if churn_risk == 'High Risk' else 'العميل مستقر حالياً'}")
            except Exception as e:
                st.error(f"❌ فشل الموديل في معالجة الـ DataFrame داخلياً: {str(e)}")

with tab2:
    st.header("Financial Feasibility & Business ROI")
    st.markdown("This section simulates the business value of using this predictive AI model vs standard operations.")
    
    c1, c2 = st.columns(2)
    with c1:
        avg_customer_value = st.number_input("Average Monthly Revenue per User (ARPU)", value=150.0, key="fin_arpu")
        total_churn_customers = st.number_input("Simulated Churn Customers Count", value=1000, key="fin_count")
    with c2:
        retention_cost = st.number_input("Cost of Retention Offer per Customer", value=30.0, key="fin_cost")
        model_accuracy_retention = st.slider("Estimated Model Success Rate in Retention (%)", 0, 100, 70)
        
    total_lost_revenue = total_churn_customers * avg_customer_value
    saved_customers = int(total_churn_customers * (model_accuracy_retention / 100))
    revenue_saved = saved_customers * avg_customer_value
    total_campaign_cost = total_churn_customers * retention_cost
    net_profit_roi = revenue_saved - total_campaign_cost
    roi_percentage = (net_profit_roi / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
    
    st.markdown("### 📊 Simulated Business Impact")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Potential Revenue Loss", f"${total_lost_revenue:,.2f}")
    kpi2.metric("Revenue Saved by AI Model", f"${revenue_saved:,.2f}", f"+{saved_customers} Customers")
    kpi3.metric("Net Profit / ROI Contribution", f"${net_profit_roi:,.2f}", f"{roi_percentage:.1f}% ROI")