import streamlit as st
import pandas as pd
import joblib

# ===== LOAD MODEL =====
pipeline = joblib.load("model.pkl")

# ===== FIX SimpleImputer ERROR =====
from sklearn.impute import SimpleImputer

def fix_pipeline(obj):
    try:
        if isinstance(obj, SimpleImputer):
            if not hasattr(obj, "_fill_dtype"):
                obj._fill_dtype = None

        if hasattr(obj, "steps"):  # Pipeline
            for name, step in obj.steps:
                fix_pipeline(step)

        if hasattr(obj, "transformers"):  # ColumnTransformer
            for name, trans, cols in obj.transformers:
                fix_pipeline(trans)

    except:
        pass

fix_pipeline(pipeline)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton>button {
    background: linear-gradient(90deg,#38bdf8,#22c55e);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.metric-card {
    background: rgba(255,255,255,0.07);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.title("📊 Dashboard")
st.sidebar.info("AI Placement Predictor")

# ===== HEADER =====
st.title("🎓 Placement Prediction Dashboard")

# ===== FORM =====
with st.form("form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 60)
        ssc = st.number_input("10th %", 0.0, 100.0)
        internships = st.number_input("Internships", 0, 10)

    with col2:
        hsc = st.number_input("12th %", 0.0, 100.0)
        degree = st.number_input("Degree %", 0.0, 100.0)
        projects = st.number_input("Projects", 0, 20)

    with col3:
        mba = st.number_input("MBA %", 0.0, 100.0)
        work_exp = st.number_input("Work Exp (months)", 0, 60)
        certifications = st.number_input("Certifications", 0, 20)

    st.subheader("👤 Personal")
    col4, col5 = st.columns(2)

    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])
        city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    with col5:
        specialization = st.selectbox("Specialization", ["Mkt&HR", "Mkt&Fin"])
        degree_field = st.selectbox("Degree Field", ["Sci&Tech", "Comm&Mgmt", "Others"])

    st.subheader("🧠 Skills")
    tech = st.slider("Technical Skills", 0, 100)
    soft = st.slider("Soft Skills", 0, 100)
    aptitude = st.slider("Aptitude", 0, 100)
    communication = st.slider("Communication", 0, 100)

    st.subheader("🏆 Extra")
    col6, col7, col8 = st.columns(3)

    with col6:
        leadership = st.number_input("Leadership Roles", 0, 5)
    with col7:
        extra = st.number_input("Extracurricular", 0, 10)
    with col8:
        backlogs = st.number_input("Backlogs", 0, 10)

    submit = st.form_submit_button("🚀 Predict Placement")

# ===== PREDICTION =====
if submit:
    try:
        data = pd.DataFrame([{
            'gender': gender,
            'city_tier': city,
            'ssc_board': 'CBSE',
            'hsc_board': 'CBSE',
            'hsc_stream': 'Science',
            'degree_field': degree_field,
            'specialization': specialization,
            'age': age,
            'ssc_percentage': ssc,
            'hsc_percentage': hsc,
            'degree_percentage': degree,
            'mba_percentage': mba,
            'internships_count': internships,
            'projects_count': projects,
            'certifications_count': certifications,
            'technical_skills_score': tech,
            'soft_skills_score': soft,
            'aptitude_score': aptitude,
            'communication_score': communication,
            'work_experience_months': work_exp,
            'leadership_roles': leadership,
            'extracurricular_activities': extra,
            'backlogs': backlogs
        }])

        # ===== SAFE PREDICTION =====
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(data)[0][1] * 100
        else:
            proba = pipeline.predict(data)[0] * 100

        # ===== METRICS =====
        colA, colB, colC = st.columns(3)

        colA.metric("🎯 Placement Chance", f"{round(proba,2)}%")
        colB.metric("💼 Experience Score", f"{work_exp} months")
        colC.metric("📊 Skill Score", f"{tech}")

        # ===== PROGRESS =====
        st.progress(int(proba))

        # ===== CHART =====
        chart_data = pd.DataFrame({
            "Category": ["Placed", "Not Placed"],
            "Value": [proba, 100 - proba]
        })

        st.subheader("📊 Prediction Breakdown")
        st.bar_chart(chart_data.set_index("Category"))

        # ===== RESULT =====
        if proba > 50:
            st.success("✅ High chances of placement")
        else:
            st.error("❌ Low chances of placement")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
