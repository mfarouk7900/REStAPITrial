import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Predictor")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("💼 Business Features")
    montant = st.number_input("Montant (Recharge)", min_value=0.0, value=0.0, step=100.0)
    frequence_rech = st.number_input("Frequence Rech", min_value=0.0, value=0.0, step=1.0)
    revenue = st.number_input("Revenue", min_value=0.0, value=0.0, step=100.0)
    arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=0.0, step=100.0)
    frequence = st.number_input("Frequence", min_value=0.0, value=0.0, step=1.0)
    data_volume = st.number_input("Data Volume", min_value=0.0, value=0.0, step=500.0)
    on_net = st.number_input("On Net", min_value=0.0, value=0.0, step=10.0)
    orange = st.number_input("Orange", min_value=0.0, value=0.0, step=10.0)
    tigo = st.number_input("Tigo", min_value=0.0, value=0.0, step=10.0)
    regularity = st.slider("Regularity", min_value=1, max_value=30, value=1)
    freq_top_pack = st.number_input("Freq Top Pack", min_value=0.0, value=0.0, step=1.0)

with col2:
    st.header("📡 Technical Features")
    region_tower_count = st.number_input("Region Tower Count", min_value=0.0, value=0.0, step=1.0)
    region_avg_range = st.number_input("Region Avg Range", min_value=0.0, value=0.0, step=10.0)
    region_avg_samples = st.number_input("Region Avg Samples", min_value=0.0, value=0.0, step=10.0)
    region_coverage_index = st.number_input("Region Coverage Index", min_value=0.0, value=0.0, step=0.1)
    region_network_quality_score = st.number_input("Region Network Quality Score", min_value=0.0, value=0.0, step=0.1)
    arr_network_quality_score = st.number_input("Arrondissement Network Quality Score", min_value=0.0, value=0.0, step=0.1)

st.markdown("---")

if st.button("Predict Churn Risk", type="primary"):
    payload = {
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
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        if response.status_code == 200 and "churn_probability" in result:
            prob = result["churn_probability"]
            risk = result["churn_risk"]
            action = result["action"]
            
            st.metric(label="Churn Probability", value=f"{prob:.2%}")
            
            if risk == "High Risk":
                st.error(f"⚠️ **Prediction:** {risk}")
                st.warning(f"💡 {action}")
            else:
                st.success(f"✅ **Prediction:** {risk}")
                st.info(f"💡 {action}")
        else:
            st.error(f"❌ {result.get('detail', 'خطأ في معالجة المدخلات')}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ السيرفر طافي. شغل api.py الأول.")