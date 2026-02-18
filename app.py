import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import os
import random

st.set_page_config(
    page_title="Spam Detector | MLOps",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;}
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        vec = joblib.load('models/vectorizer.pkl')
        mod = joblib.load('models/model.pkl')
        return vec, mod
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

vectorizer, model = load_resources()
stop_words, lemmatizer = setup_nltk()

def preprocessing_pipeline(text):
    if not isinstance(text, str): return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

if 'email_content' not in st.session_state:
    st.session_state.email_content = ""

def set_text(text):
    st.session_state.email_content = text

menu = ["🏠 System Overview", "🧠 Classification", "📊 Model Analytics"]
choice = st.sidebar.radio("Menu", menu)
st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 Author: Wiktor Pieprzowski")
st.sidebar.info("📧 Mail: zoltamordemuzrob@gmail.com")
st.sidebar.info("🐱‍👤 GitHub: ChadThunderhub")
st.sidebar.info("🚀 Stack: Streamlit, Scikit-Learn, NLTK")
st.sidebar.info("🔧 Deployment: Docker + CI/CD")

if choice == "🏠 System Overview":
    st.title("🤖 Spam/Ham Classifier")
    st.markdown("### NLP Machine Learning Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("ℹ️ **Project Architecture**")
        st.markdown("""
        This system performs **binary text classification** for automated threat detection (Spam/Phishing/Scam).
        
        Leveraging advanced **Natural Language Processing (NLP)**:
        * **Preprocessing:** Noise reduction, WordNet lemmatization, stop-words filtering.
        * **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) embedding.
        * **ML Engine:** Trained on a comprehensive dataset of **5,796 messages**, identifying complex semantic fraud patterns.
        """)
        
        st.success("📈 **Dataset Enrichment**")
        st.write("""        
        The model was trained on a **highly diversified dataset** containing consumer spam (fake lotteries, illicit ads) 
        as well as targeted phishing attacks. This robust corpus ensures real-world, diverse performance.
        """)

    with col2:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjEx/3oKIPnAiaMCws8nOsE/giphy.gif", caption="AI vs Spam")


elif choice == "🧠 Classification":
    st.title("🧠 Active Payload Scanner")
    
    if model is None:
        st.error("Model artifacts missing. Please check the 'models/' directory.")
    else:
        col_input, col_result = st.columns([2, 1])

        with col_input:
            st.subheader("Enter payload or generate sample")
            
            spam_pool = [
                "CONGRATULATIONS! You have been selected as a winner of $1,000,000. CLICK HERE to claim your prize now! No catch.",
                "URGENT: Your bank account has been locked. Please verify your identity immediately to restore access. Click this link.",
                "HOT singles in your area are waiting for you! Sign up for FREE tonight. No credit card required.",
                "Buy VIAGRA generic online, cheap price, guaranteed satisfaction. Fast shipping worldwide.",
                "Get rich quick! Crypto investment opportunity of a lifetime. Double your money in 24 hours. Guaranteed returns."
                "THANK YOU FOR YOUR ORDER. Your subscription to Premium Cloud Services has been auto-renewed for another year. A charge of $499.99 will be deducted from your credit card today. If you did not authorize this purchase, please click the attachment to cancel the subscription and request a full refund immediately.",
                "SECURITY ALERT: We detected an unauthorized login attempt on your bank account from an unrecognized device in Russia. For your safety, your access has been temporarily restricted. Please click the link below to verify your identity and restore your account immediately. Failure to act within 24 hours will result in permanent account suspension.",
                "OFFICIAL NOTIFICATION: We are pleased to inform you that your email address has been selected as the grand winner of the International Global Lottery. You have won a cash prize of $2,500,000. To claim your prize, please reply to this email with your full name and banking details. This offer is valid for a limited time only.",
                "URGENT REQUEST: I am currently in a meeting and cannot take calls. I need you to process an urgent wire transfer to a new vendor immediately. Please reply to this email so I can send you the invoice and banking details. This payment must be processed before the end of the day. Treat this with high priority.",
                "Do you suffer from low energy or performance issues? Our new clinically proven formula guarantees satisfaction and improved health. Buy generic supplements online at a cheap price. No prescription needed. Fast shipping worldwide. Click here to browse our catalog and get a 50% discount today."
            ]
            
            ham_pool = [
                "The meeting has been rescheduled to Monday morning. Please review the attached agenda beforehand.",
                "Vince, I have attached the contract for your review. Let me know if everything looks correct.",
                "Are we still on for lunch today? I have the documents you asked for.",
                "Please find the spreadsheet attached. I finished the analysis yesterday.",
                "Going to the office tomorrow. Do you need anything from the archives?",
                "The strategy meeting has been rescheduled to Monday morning in the Houston office. Please review the attached agenda and prepare your reports beforehand.",
                "Vince, I have attached the gas transportation contract for your final review. Please let me know if the figures look correct before we pass it to legal.",
                "Are we still on for lunch today to discuss the new project? I have the documents you asked for and I printed the analysis.",
                "Kindly find the spreadsheet attached regarding the risk management assessment. I finished the analysis yesterday as requested by the board.",
                "I am going to the main office tomorrow to check the archives. Do you need any specific files or contracts brought back for the meeting?",
            ]

            col_rand1, col_rand2 = st.columns(2)
            
            with col_rand1:
                if st.button("🎲 Generate SPAM Payload"):
                    st.session_state.email_content = random.choice(spam_pool)
            
            with col_rand2:
                if st.button("🎲 Generate HAM Payload"):
                    st.session_state.email_content = random.choice(ham_pool)

            user_input = st.text_area("Content:", value=st.session_state.email_content, height=200)
            
            if st.button("SCAN PAYLOAD", type="primary"):
                if user_input:
                    clean_text = preprocessing_pipeline(user_input)
                    vec_input = vectorizer.transform([clean_text]).toarray()

                    try:
                        proba = model.predict_proba(vec_input)[0]
                        spam_prob = proba[1]
                        ham_prob = proba[0]
                    except AttributeError:
                        st.error("Fatal Error: Loaded model lacks probability estimation (probability=True).")
                        st.stop()
                    
                    is_spam = spam_prob > 0.5

                    with col_result:
                        st.markdown("### Scan Results")
                        if is_spam:
                            st.error("🚨 **SPAM DETECTED**")
                            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjRiYTdtZnplajB1bjAxenFhbWJjdGg3Nnk3OHFlZGx4a2UzOTMzaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Hae1NrAQWyKA/giphy.gif")
                        else:
                            st.success("✅ **SAFE: HAM**")
                            st.image("https://media.giphy.com/media/111ebonMs90YLu/giphy.gif")
                        
                        st.markdown("---")
                        st.metric("Model Confidence", f"{max(spam_prob, ham_prob)*100:.1f}%")

                        chart_data = pd.DataFrame({
                            "Ham": [ham_prob],
                            "Spam": [spam_prob]
                        })
                        st.bar_chart(chart_data, color=["#66b3ff", "#ff4d4d"])
                        
                        with st.expander("🔍 Inspect Model Vision (NLP)"):
                            st.text("Cleaned tokens vector:")
                            st.caption(clean_text)
                else:
                    st.warning("EMPTY PAYLOAD!")

elif choice == "📊 Model Analytics":
    st.title("📊 Training Metrics & Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix", "☁️ NLP Analysis", "🏆 Model Leaderboard"])
    
    with tab1:
        st.subheader("Confusion Matrix")
        if os.path.exists('assets/confusion_matrix.png'):
            st.image('assets/confusion_matrix.png', caption='Classification performance on test dataset')
        else:
            st.warning("Missing artifact: assets/confusion_matrix.png")

    with tab2:
        st.header("🔍 Linguistic Analysis & Semantic Patterns")
        st.markdown("""     
        This section visualizes how the model classifies text. Using Natural Language Processing, 
        we extract key feature vectors that most strongly differentiate benign messages from malicious ones.
        """)
        
        st.markdown("### ☁️ Cloud of Words")
        st.write("Comparison of the most frequent tokens in both classes. Notice the distinct difference in tone and vocabulary.")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("**HAM**")
            st.caption("A mix of operational and technical communication. Words like `linux`, `user group`, `problem` indicate specialized discussions. The rest consists of pragmatic workplace language: `need`, `work`, `time`, `said`.")
   
        with row1_col2:
            st.markdown("**SPAM**")
            st.caption("Dominance of HTML tags (`td`, `tr`, `width`, `arial`) in the SPAM cloud stems from the fact that unsolicited messages often contain heavy HTML/CSS formatting (newsletters, graphic ads), unlike simple text-based HAM messages.")
                
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            if os.path.exists('assets/cloudOfWords_HAM.png'): 
                st.image('assets/cloudOfWords_HAM.png', use_container_width=True)
            else:
                st.warning("Missing artifact: assets/cloudOfWords_HAM.png")

        with row2_col2:
            if os.path.exists('assets/cloudOfWords_SPAM.png'): 
                st.image('assets/cloudOfWords_SPAM.png', use_container_width=True)
            else:
                st.warning("Missing artifact: assets/cloudOfWords_SPAM.png")

        st.markdown("---")
        
        st.subheader("📊 Feature Importance (Interpretability)")
        st.markdown("""Displays the **top 20 decision-weight tokens**. These are the vectors the model considers the strongest indicators of spam.""")
        
        col_feat1, col_feat2 = st.columns([2, 1])
        
        with col_feat1:
            if os.path.exists('assets/top_words_4_spam.png'):
                st.image('assets/top_words_4_spam.png', caption="Feature weights for the SPAM class", use_container_width=True)
            else:
                st.warning("Missing artifact: assets/top_words_4_spam.png")
        
        with col_feat2:
                st.info("💡 **Analytical Insights**")
                st.markdown("""
                1. **Aggressive CTA (Call to Action):** The model assigns massive weight to action-forcing verbs: `click` (2.16), `please` (1.98), `remove`, `order`. Spammers rely heavily on interaction.
                2. **Financial Hooks:** Classic bait tokens: `money`, `free`, `credit`, `fund`, `offer`. Confirms the majority of threats are financially motivated.
                3. **Technical & HTML Artifacts:** * `facearial`, `wi`, `style` - residual vectors from aggressive HTML/CSS formatting.
                    * `spamassassin...` - the model identified mailing list metadata signatures, effectively isolating the threat source.
                """)
                
        st.markdown("---")
        
        st.markdown("### ⚙️ SVM Optimization (Hyperparameters)")
        col_tune_img, col_tune_txt = st.columns([2, 1])
        
        with col_tune_img:
            if os.path.exists('assets/hyperparameter_tuning.png'):
                st.image('assets/hyperparameter_tuning.png', use_container_width=True)
        
        with col_tune_txt:
            st.success("🎯 **Regularization Impact**")
            st.markdown("""
            Various values for the **C parameter** (Regularization) were tested. 
            A higher **C** tells the model to prioritize correct classification 
            of training points, while a lower **C** encourages a smoother 
            decision boundary. 
            
            The plot shows the **sweet spot** where we achieve maximum **F1-Score** without over-fitting.
            """)

    with tab3:
        st.subheader("🏆 Model Leaderboard")
        st.write("Cross-validation results captured during the training phase.")
        
        if os.path.exists('assets/model_results.csv'):
            df_results = pd.read_csv('assets/model_results.csv')
            
            df_results = df_results.round(4)
            
            st.dataframe(
                df_results.style.highlight_max(axis=0, subset=['F1-Score', 'Accuracy'], color="#22970d"),
                use_container_width=True,
                hide_index=True
            )
            
            best_model = df_results.iloc[0]['Model']
            best_f1 = df_results.iloc[0]['F1-Score']
            st.success(f"**🥇 Top Performing Engine:** {best_model} (F1: {best_f1})")
            
        else:
            st.error("Missing artifact: assets/model_results.csv")
            
        st.subheader("📊 F1-Score Benchmark")
        
        col_chart, col_info = st.columns([2, 1]) # Wykres szerszy (2), opis węższy (1)
        
        with col_chart:
            if os.path.exists('assets/f1_score_comparison.png'):
                st.image('assets/f1_score_comparison.png', caption="Model F1-Score Benchmark", use_container_width=True)
            else:
                st.warning("Missing artifact: assets/f1_score_comparison.png (Run the training script first)")
        
        with col_info:
            st.info("ℹ️ **Why F1-Score?**")
            st.markdown("""
            In Spam Detection, **Accuracy can be misleading** (since most emails are usually safe/Ham).
            
            **F1-Score** is the harmonic mean of Precision and Recall. It ensures that the model:
            1. Doesn't miss Spam (**High Recall**)
            2. Doesn't falsely flag safe emails (**High Precision**)
            
            The chart visualizes which algorithm balances these two critical metrics best.
            """)
        
st.markdown("---")
st.markdown("© 2026 | Developed by Wiktor Pieprzowski | Automated CI/CD MLOps Pipeline")