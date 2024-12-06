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
st.set_page_config(page_title="CardioBuddy", page_icon="💓", layout="wide")

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
    st.title("💓 Welcome to CardioBuddy")
    st.write("This app provides ECG analysis, an AI chatbot for healthcare consultation, and advanced ECG report generation using Generative AI.")

# ECG Analysis Page
elif nav == "ECG Analysis":
    st.title("📊 ECG Analysis")
    uploaded_image = st.file_uploader("Upload an ECG Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        pred_class = pred_and_plot(ecg_model, image, class_names)
        st.write(pred_class)

# Chatbot Page
elif nav == "Chatbot":
    st.title("🤖 AI Chatbot")
    st.write("Ask anything or consult the chatbot for healthcare-related queries.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Your Query:")   

    if st.button("Send") and user_input:
        response = get_gemini_response(user_input, st.session_state['chat_history'])
        st.session_state['chat_history'].append(("User", user_input))  # Corrected the typo from 'input' to 'user_input'
    
        for chunk in response:
            st.session_state['chat_history'].append(("Bot", chunk.text))

    st.subheader("History")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

# Gen AI Page
elif nav == "Gen AI":
    st.title("🧠 Generate ECG Report with Generative AI")
    
    # Language selection specific to this page with English as the default
    languages = [
        "English", "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada", 
        "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", 
        "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil", 
        "Telugu", "Urdu"
    ]
    lang = st.selectbox("Select Language for Report", options=languages, index=0)  # 'index=0' sets English as default

    # File uploader for ECG image
    ecg_image = st.file_uploader("Upload ECG Image for Analysis", type=["png", "jpg", "jpeg"])
    
    if ecg_image is not None:
        st.image(ecg_image, caption="Uploaded ECG Image", use_column_width=True)
        
        # Generate report button
        if st.button("Generate ECG Report"):
            with st.spinner("Analyzing ECG image..."):
                ecg_details = generate_ecg_details(ecg_image,lang)
            
            st.header("Generated ECG Report")
            st.markdown(ecg_details)
            
            # Create a downloadable document file
            doc_file_stream = create_doc(ecg_details, ecg_image)
            st.download_button(
                label="Download ECG Report",
                data=doc_file_stream,
                file_name=f"ECG_Report_{lang}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

