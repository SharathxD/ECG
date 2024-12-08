import os
import tensorflow as tf
from datetime import datetime
from PIL import Image
from io import BytesIO
from docx import Document
from docx.shared import Inches
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
genai_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Load the ECG model
ecg_model = tf.keras.models.load_model('ecg.h5')
chat = genai_model.start_chat(history=[])

# ECG classes
class_names = [
    'ECG Images of Myocardial Infarction Patients (240x12=2880)',
    'ECG Images of Patient that have History of MI (172x12=2064)',
    'ECG Images of Patient that have abnormal heartbeat (233x12=2796)',
    'Normal Person ECG Images (284x12=3408)'
]

# Page Config
st.set_page_config(page_title="CardioBuddy", page_icon="🫀", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        color: #2c3e50;
        background-color: #f0f3f6;
    }
    .main {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #3498db;
    }
    .feature-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        color: #3498db;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTextInput>div>div>input {
        border-color: #3498db;
    }
    .stTextInput>div>div>input:focus {
        border-color: #2980b9;
        box-shadow: 0 0 0 1px #2980b9;
    }
    .stSelectbox>div>div>div {
        border-color: #3498db;
    }
    .stSelectbox>div>div>div:focus {
        border-color: #2980b9;
        box-shadow: 0 0 0 1px #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def get_gemini_response(input,history):
    prompt=f"""Initial Configuration

1. Role: Medical Assistant (CVD Specialist)
2. Knowledge Domain: Cardiovascular diseases, symptoms, diagnosis, treatment, and prevention.
3. Conversational Scope: Patient consultation, medical inquiry, and education.
Respond as a medical professional specializing in cardiovascular diseases. Address the user's query with accurate and relevant information. Provide clear explanations, diagnosis considerations, and treatment options. Limit responses to medical aspects only.

User Query: {input}

Generate a comprehensive response following these guidelines:

Response Guidelines

1. Accuracy: Ensure responses are evidence-based and up-to-date.
2. Clarity: Use simple, non-technical language.
3. Relevance: Focus on cardiovascular diseases.
4. Professionalism: Maintain a neutral, empathetic tone.
5. Contextual understanding: Consider patient history, symptoms, and medical context.

Output Requirements

1. Format: Structured response with headings and bullet points.
2. Length: Concise, ideally 200-300 words.
3. Tone: Professional, empathetic.

Example Response

For user query: 'What are symptoms of heart failure?'

Response:

Symptoms
- Shortness of breath (dyspnea)
- Fatigue
- Swelling in legs, ankles, and feet (edema)
- Rapid or irregular heartbeat
- Coughing up pink, frothy mucus

Medical Considerations
If experiencing symptoms, consult a healthcare provider for proper diagnosis and treatment.You are an AI assistant . Here's the conversation so far:\n\n
            {''.join([f'{role}:{text}'for role, text in history])}\n\n
            now, Here's the user's new query:{input}\n\n
            Please provide a comprehensive and informative response"""
    response=chat.send_message(prompt,stream=False)
    return response

def load_and_prep_image(image, img_shape=224):
    img = tf.convert_to_tensor(image)
    img = tf.image.decode_image(tf.io.encode_jpeg(img), channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.0
    return img

def pred_and_plot(model, image, class_names):
    img = load_and_prep_image(image)
    pred = model.predict(tf.expand_dims(img, axis=0))
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    return pred_class

def generate_ecg_details(ecg_image,lang):
    current_date = datetime.now().strftime('%Y-%m-%d')
    prompt = f"""
    Analyze the provided ECG image and generate a comprehensive, structured report in this {lang} language only which includes:

    Patient Information

1. Patient's vital signs (if identifiable): heart rate, blood pressure, respiratory rate, and oxygen saturation.
2. Patient demographics (if available): age, sex, and medical history.

ECG Analysis

1. Heart rate: calculate and report the average heart rate in beats per minute (bpm).
2. Rhythm analysis: identify and describe the rhythm (e.g., sinus, atrial fibrillation, ventricular tachycardia).
3. Interval measurements: report PR, QRS, and QT intervals.
4. Potential abnormalities: identify and describe any notable features, such as:
- Arrhythmias (e.g., PVCs, PACs)
- Conduction disturbances (e.g., AV block, bundle branch block)
- Ischemic changes (e.g., ST-segment elevation/depression)
- Other notable findings (e.g., Wolff-Parkinson-White syndrome)

Medical Considerations

1. Suggested next steps: recommend further testing, monitoring, or medical interventions.
2. Differential diagnoses: provide a list of possible conditions based on ECG findings.
3. Clinical implications: discuss potential risks and consequences of identified abnormalities.

Report Requirements

1. Report should be clear, concise, and free of technical jargon.
2. Include relevant medical terminology and abbreviations.
3. Provide a confident and evidence-based assessment.
4. Report should be dated with the current date.

Generatated on {current_date}
    
"""
    chat_session = genai_model.start_chat(history=[])
    response = chat_session.send_message([prompt])
    return response.text

def create_doc(report_text, ecg_image):
    doc = Document()
    doc.add_heading('ECG ANALYSIS REPORT', 0)
    
    for line in report_text.split("\n"):
        if line.strip():
            if line.startswith("**") and line.endswith("**"):
                doc.add_heading(line.strip("**"), level=1)
            elif line.startswith("-"):
                doc.add_paragraph(line.strip(), style="List Bullet")
            else:
                doc.add_paragraph(line.strip())

    doc.add_heading('ECG Tracing:', level=1)
    image_stream = BytesIO(ecg_image.getvalue())
    doc.add_picture(image_stream, width=Inches(6))

    file_stream = BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream

# Sidebar Navigation
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to", ["Home", "ECG Analysis", "Chatbot", "Gen AI"])

# Home Page
if nav == "Home":
    st.markdown("<h1 style='text-align: center; color: #3498db;'>🫀 Welcome to CardioBuddy</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI-powered cardiology assistant</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3><span class='feature-icon'>🔬</span> ECG Analysis</h3>
            <p>Upload ECG images for instant AI-powered analysis and classification.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <h3><span class='feature-icon'>🤖</span> AI Chatbot</h3>
            <p>Consult our AI for healthcare-related queries and get evidence-based responses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3><span class='feature-icon'>🧠</span> Generative AI Reports</h3>
            <p>Generate comprehensive ECG reports using cutting-edge AI technology.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='feature-card'>
            <h3><span class='feature-icon'>📊</span> Downloadable Reports</h3>
            <p>Create and download detailed ECG analysis reports in DOCX format.</p>
        </div>
        """, unsafe_allow_html=True)

# ECG Analysis Page
elif nav == "ECG Analysis":
    st.markdown("<h1 style='text-align: center; color: #3498db;'>🔬 ECG Analysis</h1>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an ECG Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing ECG..."):
            pred_class = pred_and_plot(ecg_model, image, class_names)
        st.success(f"Analysis Result: {pred_class}")

# Chatbot Page
elif nav == "Chatbot":
    st.markdown("<h1 style='text-align: center; color: #3498db;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask anything or consult the chatbot for healthcare-related queries.</p>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Your Query:")   

    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            response = get_gemini_response(user_input, st.session_state['chat_history'])
        st.session_state['chat_history'].append(("User", user_input))
    
        for chunk in response:
            st.session_state['chat_history'].append(("Bot", chunk.text))

    st.subheader("Chat History")
    for role, text in st.session_state['chat_history']:
        st.markdown(f"**{role}**: {text}")

# Gen AI Page
elif nav == "Gen AI":
    st.markdown("<h1 style='text-align: center; color: #3498db;'>🧠 Generate ECG Report with AI</h1>", unsafe_allow_html=True)
    
    languages = [
        "English", "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada", 
        "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", 
        "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil", 
        "Telugu", "Urdu"
    ]
    lang = st.selectbox("Select Language for Report", options=languages, index=0)

    ecg_image = st.file_uploader("Upload ECG Image for Analysis", type=["png", "jpg", "jpeg"])
    
    if ecg_image is not None:
        st.image(ecg_image, caption="Uploaded ECG Image", use_column_width=True)
        
        if st.button("Generate ECG Report"):
            with st.spinner("Analyzing ECG image..."):
                ecg_details = generate_ecg_details(ecg_image,lang)
            
            st.markdown("<h2 style='text-align: center;'>Generated ECG Report</h2>", unsafe_allow_html=True)
            st.markdown(ecg_details)
            
            doc_file_stream = create_doc(ecg_details, ecg_image)
            st.download_button(
                label="Download ECG Report",
                data=doc_file_stream,
                file_name=f"ECG_Report_{lang}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

