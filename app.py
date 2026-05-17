import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

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
        customer_payload = {
            "montant": float(montant),
            "frequence_rech": float(frequence_rech),
            "revenue": float(revenue),
            "arpu_segment": float(arpu_segment),
            "frequence": float(frequence),
            "data_volume": float(data_volume),
            "on_net": float(on_net),
            "orange": float(orange),
            "tigo": float(tigo),
            "regularity": float(regularity),
            "freq_top_pack": float(freq_top_pack),
            "region_tower_count": float(region_tower_count),
            "region_avg_range": float(region_avg_range),
            "region_avg_samples": float(region_avg_samples),
            "region_coverage_index": float(region_coverage_index),
            "region_network_quality_score": float(region_network_quality_score),
            "arr_network_quality_score": float(arr_network_quality_score)
        }
        
        api_url = "http://127.0.0.1:8000/predict"
        
        try:
            with st.spinner("جاري الاتصال بـ REST API وحساب التوقع بالنموذج..."):
                response = requests.post(api_url, json=customer_payload)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data and "prediction" in response_data:
                    result = response_data["prediction"]
                    
                    st.markdown("---")
                    if result["churn_risk"] == "High Risk":
                        st.error(f"⚠️ **Prediction: {result['churn_risk']}**")
                    else:
                        st.success(f"✅ **Prediction: {result['churn_risk']}**")
                        
                    st.metric(label="Churn Probability", value=f"{round(result['churn_probability'] * 100, 2)}%")
                    st.info(f"💡 **Suggested Corporate Action:** {result['suggested_action']}")
                else:
                    st.error("الـ API أرجع استجابة بتنسيق غير متوقع.")
            else:
                error_detail = response.json().get('detail', 'خطأ غير معروف')
                st.error(f"❌ خطأ من سيرفر الـ API: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ فشل الاتصال بالـ REST API. تأكد من تشغيل أمر uvicorn api:app --reload أولاً!")

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