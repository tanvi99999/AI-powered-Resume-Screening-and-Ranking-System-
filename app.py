import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set modern page config
st.set_page_config(
    page_title="AI Resume Screener", 
    page_icon="📄", 
    layout="centered"
)

# Custom CSS for minimalistic design
st.markdown("""
    <style>
        .css-18e3th9 { padding-top: 2rem; }
        .stTextArea textarea { font-size: 16px; }
        .stFileUploader > div { border-radius: 10px; padding: 10px; }
        .stDataFrame { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():  
            text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

# ---- UI DESIGN ----
st.title("📄 AI Resume Screener")
st.write("An AI-powered tool to rank resumes based on job relevance.")

# Job description input
st.subheader("📝 Job Description")
job_description = st.text_area("Enter the job description here...", height=150)

# File uploader
st.subheader("📂 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

# Process resumes if both job description & files are provided
if uploaded_files and job_description:
    st.subheader("📊 Ranking Resumes")

    with st.spinner("Analyzing resumes... ⏳"):
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

    # Show results with color-coded progress bars
    for index, row in results.iterrows():
        st.write(f"📌 **{row['Resume']}**")
        st.progress(float(row['Score']))  # Visual progress bar

    # Display results in a clean table
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    st.dataframe(results)  # ✅ Use st.dataframe() to prevent extra text output

    # Download results
    st.download_button(
        label="📥 Download Results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="resume_rankings.csv",
        mime="text/csv"
    )
