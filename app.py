import streamlit as st
from rag_engine import get_assessment_recommendation

st.set_page_config(
    page_title="SHL GenAI Assessment Recommender",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar
with st.sidebar:
    st.title("🔧 Settings")
    st.markdown("Choose the number of recommendations:")
    top_k = st.slider("Top-K Recommendations", 1, 10, 3)

# Header
st.markdown("""
    <div style="text-align: center;">
        <h1>🧠 SHL Assessment Recommender</h1>
        <p style="font-size: 18px;">Paste a job description and get the best-matched SHL assessments using AI.</p>
    </div>
""", unsafe_allow_html=True)

# Job description input
job_description = st.text_area("📄 Job Description", height=300, placeholder="Paste job description here...")

# Submit button
if st.button("🚀 Recommend Assessments"):
    if job_description.strip():
        with st.spinner("Analyzing and retrieving top SHL assessments..."):
            results = get_assessment_recommendation(job_description, top_k=top_k)

        st.success("✅ Recommendations Ready!")
        for i, assessment in enumerate(results, start=1):
            st.markdown(f"""
                <div style="padding:10px;margin-bottom:10px;border-left:5px solid #4CAF50;background:#f9f9f9">
                    <h4>{i}. {assessment['title']}</h4>
                    <p>{assessment['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid job description above.")

