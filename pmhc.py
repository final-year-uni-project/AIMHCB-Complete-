import os, logging, time, smtplib, email.mime.text, email.mime.multipart, hashlib, cv2
import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from fer import FER

# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    groq_client = None
    logging.warning(f"Groq client initialization failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Crisis keywords for detection
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
    'no point living', 'can\'t go on', 'end it all', 'hurt myself', 'self harm',
    'cutting', 'cut myself', 'want to cut', 'going to cut', 'overdose', 'jump off',
    'hang myself', 'worthless', 'hopeless', 'can\'t take it anymore', 'can\'t take this anymore',
    'nobody cares', 'everyone would be better without me', 'want to kill myself',
    'going to kill myself', 'self-harm', 'self injury', 'harm myself', 'injure myself', 'bleeding'
]

# Simple user database (in production, use a proper database)
USER_DB_FILE = r"data\users.txt"
PROFILE_DIR = r"data\profiles"

# Ensure the directory exists
os.makedirs(PROFILE_DIR, exist_ok=True)

def save_user_profile(username, profile_text):
    """
    Save user's personal profile (pp) to a file
    """
    profile_file = os.path.join(PROFILE_DIR, f"{username}_profile.txt")
    try:
        with open(profile_file, "w", encoding="utf-8") as f:
            f.write(profile_text)
        logging.info(f"✅ Profile saved for {username}")
    except Exception as e:
        logging.error(f"❌ Failed to save profile for {username}: {e}")

def load_user_profile(username):
    global profile_text
    """
    Load user's personal profile (pp) from a file
    """
    profile_file = os.path.join(PROFILE_DIR, f"{username}_profile.txt")
    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_text = f.read()
            return profile_text
        except Exception as e:
            logging.error(f"❌ Failed to load profile for {username}: {e}")
            return ""
    else:
        return ""

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from file"""
    users = {}
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('|')
                    if len(parts) == 5:  # Updated to handle phone number
                        username, email, name, phone, password_hash = parts
                        users[username] = {
                            'email': email,
                            'name': name,
                            'phone': phone,
                            'password_hash': password_hash
                        }
                    elif len(parts) == 4:  # Backward compatibility for existing users
                        username, email, name, password_hash = parts
                        users[username] = {
                            'email': email,
                            'name': name,
                            'phone': 'Not provided',
                            'password_hash': password_hash
                        }
    return users

def save_user(username, email, name, phone, password):
    """Save new user to file"""
    password_hash = hash_password(password)
    # Clean phone number (remove empty strings)
    phone = phone.strip() if phone else 'Not provided'
    with open(USER_DB_FILE, 'a') as f:
        f.write(f"{username}|{email}|{name}|{phone}|{password_hash}\n")

def authenticate_user(username, password):
    """Authenticate user"""
    users = load_users()
    if username in users:
        if users[username]['password_hash'] == hash_password(password):
            return users[username]
    return None

def summarize_chat_history(chat_history):
    '''Summarize older chat history using the llama/Groq model.
       Returns a string summary that can be prepended to new prompts'''
    
    # get full history
    history_text=""
    for chat in chat_history:
        history_text+=f"User: {chat.get('user', 'Unavailable')}\n Counselor: {chat.get('bot','Unavailable')}"
        
    
    system_prompt='''You are an efficient summarizing tool used to
    compress texts between a user and their counselor.
    Summarize the given text such that;
    In the user's text:
    contextual relevance is maintained.
    To the best of your ability, keep all emotional and informational content.
    In the counselor's text:
    Any questions or elements that would influence the user's response is captured such that their next response is aptly contextual by the summary.
    Your output does not have to be perfectly readable, just understandable.
    Ensure the summary captures the user’s state, AND the counselor’s supportive tone, without repeating full sentences.
    Avoid repeating supportive filler phrases (e.g., "I understand", "That must be hard") unless essential for context.
    Aim for an output equivalent or less than 100 tokens.'''
    
    messages= [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": history_text}
        ]
    with st.spinner("Summarizing.."):
        summary=call_groq_api(messages=messages, temperature=0.4)
    
    return summary
                     
    
def analyze_sentiment_and_risk(text):
    """
    Simple sentiment analysis using TextBlob and crisis keyword detection
    """
    try:
        # Simple TextBlob sentiment analysis (like your example)
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict =analyzer.polarity_scores(text)
        polarity = sentiment_dict['compound']  # -1 (negative) to 1 (positive)

        # Check for crisis keywords
        text_lower = text.lower()
        crisis_keywords_found = []
        for keyword in CRISIS_KEYWORDS:
            if keyword in text_lower:
                crisis_keywords_found.append(keyword)

        # Calculate risk score based on keywords and sentiment
        risk_score = 0

        # Crisis keywords are the primary indicator
        if crisis_keywords_found:
            risk_score += len(crisis_keywords_found) * 4  # Each keyword adds 4 points

        # Sentiment-based risk (secondary factor)
        if polarity <= -0.5:  # Very negative
            risk_score += 3
        elif polarity <= -0.2:  # Negative
            risk_score += 2
        elif polarity < 0:  # Slightly negative
            risk_score += 1

        # Mental health indicators
        mental_health_words = ['depressed', 'depression', 'anxiety', 'anxious', 'panic', 'scared', 'suicidal', 'desperate', 'overwhelmed']
        mental_health_count = sum(1 for word in mental_health_words if word in text_lower)
        if mental_health_count > 0:
            risk_score += mental_health_count

        # Cap at 10
        risk_score = min(risk_score, 10)

        # Determine crisis level
        if risk_score >= 8:
            crisis_level = "SEVERE"
        elif risk_score >= 6:
            crisis_level = "HIGH"
        elif risk_score >= 4:
            crisis_level = "MODERATE"
        else:
            crisis_level = "LOW"

        # Sentiment label based on polarity
        if polarity > 0.1:
            sentiment_label = "POSITIVE"
        elif polarity < -0.1:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"

        # Debug info for crisis detection
        if crisis_keywords_found:
            logging.debug(f"Crisis keywords: {crisis_keywords_found}, Polarity: {polarity}, Risk: {risk_score}")


        return polarity, risk_score, crisis_level, sentiment_label

    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return 0, 0, "LOW", "NEUTRAL"

def send_crisis_alert(user_email, user_name, user_message, risk_score, contact_info=None):
    """
    Send actual crisis alert email using Gmail SMTP
    """
    global smtp_email, smtp_password, counselor_email
    try:
        if not st.session_state.developer:
            counselor_email = os.getenv("COUNSELOR_EMAIL", "edmundquarshie019@gmail.com")
            smtp_email = os.getenv("SMTP_EMAIL", "edmundquarshie019@gmail.com")
            smtp_password = os.getenv("SMTP_PASSWORD")

            if not smtp_password:
                logging.error("SMTP_PASSWORD not configured in .env file")
                return False

        # Create email message
        msg = email.mime.multipart.MIMEMultipart()
        msg['From'] = smtp_email
        msg['To'] = counselor_email
        msg['Subject'] = f"MENTAL HEALTH ALERT - {user_name} - Risk: {risk_score}/10"

        body = f"""
MENTAL HEALTH CRISIS ALERT

User: {user_name} ({user_email})
Risk Score: {risk_score}/10
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

User Message:
"{user_message}"

Contact Information:
{contact_info if contact_info else 'Not provided'}

PLEASE FOLLOW UP IMMEDIATELY if this indicates a genuine crisis.

This alert was automatically generated by the Mental Health Chatbot system.
User's actual email: {user_email}
"""

        msg.attach(email.mime.text.MIMEText(body, 'plain'))

        # Send actual email using Gmail SMTP
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(smtp_email, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_email, counselor_email, text)
            server.quit()

            # Log success
            logging.info(f"✅ CRISIS EMAIL SENT: {user_name} ({user_email}) - Risk: {risk_score}/10")
            print(f"\n✅ CRISIS EMAIL SENT SUCCESSFULLY:")
            print(f"FROM: {smtp_email}")
            print(f"TO: {counselor_email}")
            print(f"USER: {user_name} ({user_email})")
            print(f"RISK: {risk_score}/10")

            # Also log to file for backup
            alert_message = f"""
=== EMAIL SENT SUCCESSFULLY ===
FROM: {smtp_email}
TO: {counselor_email}
USER: {user_name} ({user_email})
SUBJECT: {msg['Subject']}
TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}

{body}
========================
"""
            with open(r"data\crisis_alerts.log", "a") as f:
                f.write(alert_message + "\n")

            return True

        except Exception as email_error:
            logging.error(f"❌ EMAIL SENDING FAILED: {email_error}")
            print(f"\n❌ EMAIL SENDING FAILED: {email_error}")

            # Log failure for backup
            alert_message = f"""
=== EMAIL FAILED ===
ERROR: {email_error}
USER: {user_name} ({user_email})
RISK: {risk_score}/10
MESSAGE: {user_message}
TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}
========================
"""
            with open(r"data\crisis_alerts.log", "a") as f:
                f.write(alert_message + "\n")

            return False

    except Exception as e:
        logging.error(f"Crisis alert setup error: {e}")
        return False

def display_crisis_intervention(risk_score, user_message, user_email, user_name):
    """
    Display crisis intervention interface
    """
    st.error("🚨 **CRISIS ALERT DETECTED**")
    st.warning(f"**Risk Level: {risk_score}/10**")
    
    # Emergency resources
    st.markdown("### 🆘 **IMMEDIATE HELP AVAILABLE:**")
    col1, col2 = st.columns(2)
    
    ################# Make changes here ############ 
    with col1:
        st.markdown("""
        **🚨 Emergency Services:**
        - **Call 112** (Emergency Line)
        - **+233 244 846 701** (Mental Health Authority Helpline)
        - **+233 303 932 545** (Youth Helpline)
        """)
        
    with col2:
        st.markdown("""
        **💬 Online Support:**
        - **suicidepreventionlifeline.org**
        - **crisistextline.org**
        - **nami.org** (Mental Health Support)
        """)
    
    # Contact form
    st.markdown("### 📞 **Additional Support Resources**")
    st.info("If you need immediate professional help, please use the emergency contacts above.")


# Groq API client function
def call_groq_api(messages, model="llama-3.3-70b-versatile", max_tokens=1500, temperature=0.7):
    """
    Call Groq API for chat completions
    """
    try:
        if not groq_client:
            logging.error("Groq client not initialized")
            return None

        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return None

# Load mental health dataset
try:
    csv_paths = [
        r"data\AI_Mental_Health.csv",
        "../AI_Mental_Health.csv",
        "Personalized_Mental_Healthcare-Chatbot-main/AI_Mental_Health.csv"
    ]

    mentalhealth = None
    csv_path_used = None

    for csv_path in csv_paths:
        try:
            mentalhealth = pd.read_csv(csv_path)
            csv_path_used = csv_path
            break
        except FileNotFoundError:
            continue

    if mentalhealth is not None:
        knowledge_base = {}
        for i in range(len(mentalhealth)):
            question = str(mentalhealth.iloc[i]["Questions"]).lower()
            answer = str(mentalhealth.iloc[i]["Answers"])
            if question and answer and question != 'nan' and answer != 'nan':
                knowledge_base[question] = answer
    else:
        knowledge_base = {}

except Exception as e:
    knowledge_base = {}

def generate_response(input_text, fem, user_email, user_name, user_phone=None):
    """
    Generate response with sentiment analysis and crisis detection
    """
    global gradient
    with st.spinner("Thinking..."):
        try:
            # Analyze sentiment and risk
            sentiment_score, risk_score, crisis_level, sentiment_label = analyze_sentiment_and_risk(input_text)
            if risk_score<=3:
                crisis_line="light encouragement. "
            elif risk_score<=6:
                crisis_line="Very attentive, empathetic, offer gentle coping support"
            else:
                crisis_line="IMPORTANT: Kindly but firmly encourage contacting emergency services or helplines. Always include one concrete helpline (e.g.,  in the US)."
            
            if fem!=0.0 and not st.session_state.fem_acknowledged:
                st.session_state.fem_acknowledged=True
                fem_line="""- Acknowledge facial expression, gently once at the start and sparsely afterward
            (e.g., “You look a bit sad,” “You seem brighter today,” or “You look neutral”). 
            - Do not rashly bring up facial expression again unless the user explicitly mentions it.
            - Gently acknowledge facial expression, e.g., “You look a bit sad,” “You seem brighter,” or “You look neutral.”
            """
            else: fem_line="Appropriately affirm the user's 'air'. eg. User: I feel so down You: I can tell you're a little off...(with light encouragement)"
            if st.session_state.first_prompt:
                instruct=f"""Use the personal profile included at the beginning, if any, as context for your response to highlight your care for what they tell you
                Context:personal profile + user {user_name} has facial expression score {fem} (-1 to 1, 0 = neutral), 
                risk score {risk_score}/10, and sentiment {sentiment_label}."""
            else:instruct=f"""Context:chat history + user {user_name} has facial expression score {fem} (-1 to 1, 0 = neutral), 
                risk score {risk_score}/10, and sentiment {sentiment_label}."""
                
            # Generate AI response
            system_prompt = f"""
            You are a WellBot, a compassionate mental health chatbot.
            You are the Counselor represented in the context provided if any. 
            {instruct}
            
            Instructions:
            - Speak in a natural, conversational tone that matches the user’s mood.
            {fem_line} 
            - Be empathetic, supportive, and professional, never clinical or diagnostic. 
            - Avoid redundant or unnecessary questions—respond as if truly listening. 
            - Keep responses concise (≤20× user’s input length). 
            - Risk guidance:
            • {crisis_line}
            """

            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
            
            bot_response = call_groq_api(messages=messages)
            
            if bot_response:
                with st.container(border=True):
                    st.write(f"### 🤖 WellBot (for {user_name}):")
                    st.write(bot_response)
                    if risk_score >= 3:
                        st.info("💙 **Additional Resources:** If you're struggling, consider reaching out to a mental health professional or calling +233 244 846 701 for support.")
            else:
                st.error("❌ Unable to generate response. Please try again.")
            
            st.markdown("---")

            # Automatic email alert for moderate to critical risk
            if risk_score >= 4:
                # Send automatic email alert for moderate+ risk
                auto_contact_info = f"Automatic alert triggered by sentiment analysis\nRisk Level: {crisis_level}\nPhone: {user_phone if user_phone and user_phone != 'Not provided' else 'Not provided'}"
                email_sent = send_crisis_alert(user_email, user_name, input_text, risk_score, auto_contact_info)

                # if email_sent:
                #     st.warning(f"🚨 **Alert Sent**: Risk level {crisis_level} detected. Counselor has been notified automatically.")
                # else:
                #     st.error("⚠️ **Alert Failed**: Unable to send automatic alert. Please contact emergency services if needed.")

            # Crisis intervention interface for high risk
            if risk_score >= 6:
                display_crisis_intervention(risk_score, input_text, user_email, user_name)
                return
            
            # Display prominent sentiment dashboard
            st.markdown("### 📊 **Sentiment Analysis Results**")

            # Color-coded risk level display
            if risk_score >= 8:
                st.error(f"🚨 **SEVERE RISK DETECTED** - Score: {risk_score}/10")
            elif risk_score >= 6:
                st.warning(f"⚠️ **HIGH RISK DETECTED** - Score: {risk_score}/10")
            elif risk_score >= 4:
                st.warning(f"🟡 **MODERATE RISK DETECTED** - Score: {risk_score}/10")
            else:
                st.success(f"✅ **LOW RISK** - Score: {risk_score}/10")

            # Detailed metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            w,h=30,30
            with col1:
                st.metric("Risk Score", f"{risk_score}/10", delta=None)
            with col2:
                st.metric("Crisis Level", crisis_level)
            with col3:
                st.metric("Sentiment", sentiment_label)
            with col4:
                sentiment_color = "🟢" if sentiment_score > 0 else "🔴" if sentiment_score < -0.2 else "🟡"
                st.metric("Mood", f"{sentiment_color} {sentiment_score:.2f}")
                
        except Exception as e:
            logging.error(f"Error in response generation: {e}")
            st.error(f"🚨 Error: {e}")
    gradient=0.0
    return bot_response

@st.cache_resource
def load_fem_model():
    return FER(mtcnn=True)
    
def get_fem(getFacialExp, detector, photo):
    if getFacialExp and photo is not None:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_emotions(img_rgb)
        if not faces:
            st.warning("No face detected; using neutral score.")
            gradient=0.0
            return gradient
        e = faces[0]['emotions']
        emotion_array=np.array([
            e['angry'],
            e['disgust'],
            e['fear'],
            e['happy'],
            e['sad'],
            e['surprise'],
            e['neutral']
        ])
        translation_array=np.array([
            -0.7,
            -0.4,
            -0.4,
            0.8,
            -0.9,
            0.4,
            0.0
            ])
        
        gradient=round(np.dot(emotion_array, translation_array),3)
    elif getFacialExp and not photo:
        st.error("Photo could not be uploaded")
        gradient=0.0
    else:
        gradient=0.0
    if not st.session_state.developer:
        return gradient
    else:
        global emotions
        emotions=e
        return gradient
    
def generate_user_profile_summary(pp, chat_prefix):
    """
    Generates a concise 50-word profile summarizing user state.
    """
    system_prompt = f"""
    You are an assistant that creates a short, 50-word profile on a user and their mental state and interaction style 
    based on the following chat summary between them and their counselor and previous profile:
    {chat_prefix} and {pp}
    At the end of the profile include any reference the user or counselor made last to add context for their next meeting.
    Produce the concise profile in about than 50 words.
    """
      
    messages = [{"role": "system", "content": system_prompt}]
    
    profile_summary = call_groq_api(messages=messages, temperature=0.3)
    return profile_summary or "No profile available."

# Main application
def main():
    global smtp_email, smtp_password, counselor_email, profile_text, gradient
    profile_text=""
    

    st.set_page_config(
        page_title="Authenticated Mental Health Chatbot",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Authenticated Mental Health Chatbot")
    if st.session_state.get: st.subheader("Developer Mode")
    st.markdown("🔐 **Secure, Personalized AI Mental Health Support**")
    st.markdown("---")

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'fem_metric' not in st.session_state:
        st.session_state.fem_metric=False
    if 'first_prompt' not in st.session_state:
        st.session_state.first_prompt=True
    if 'pp' not in st.session_state:
        st.session_state.pp=None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    if 'developer' not in st.session_state:
        st.session_state.developer=False
    if 'fem_acknowledged' not in st.session_state:
        st.session_state.fem_acknowledged = False

    # Authentication
    if not st.session_state.authenticated:
        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Developer"],)

        with tab1:
            st.subheader("🔑 Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")

                if login_button:
                    user_info = authenticate_user(username, password)
                    if user_info:
                        st.session_state.authenticated = True
                        st.session_state.user_info = user_info
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        # Load personal profile
                        st.session_state.pp = load_user_profile(username) or ""
                        profile_text = st.session_state.pp  # global for use in chat
                        
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")

        with tab2:
            st.subheader("📝 Register")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email Address")
                new_name = st.text_input("Full Name")
                new_phone = st.text_input("Phone Number (optional)", placeholder="e.g., +1-555-123-4567")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Create Account")

                if register_button:
                    if not all([new_username, new_email, new_name, new_password]):
                        st.error("Please fill in all required fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif new_username in load_users():
                        st.error("Username already exists")
                    else:
                        try:
                            save_user(new_username, new_email, new_name, new_phone, new_password)
                            st.success("✅ Account created successfully! Please login.")
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
        
        with tab3:
            devpass= st.text_input("Password", type="password")
            if devpass.lower()=="alabi":
                st.success("✅Developer mode enabled!")
                st.session_state.developer=True
                st.session_state.user_info={'name':'Developer','email':'Void','phone':'Void'}
                st.session_state.username="Dev"
                st.session_state.authenticated=True
                st.rerun()
            else:
                st.write("Developer mode will not be enabled.")
                st.session_state.authenticated=False
                st.rerun()
                
    else:
        # User is authenticated
        user_info = st.session_state.user_info
        username = st.session_state.username

        # Sidebar with user info and logout
        with st.sidebar:
            st.write(f'Welcome **{user_info["name"]}**')
            st.write(f'Email: {user_info["email"]}')
            st.write(f'Phone: {user_info.get("phone", "Not provided")}')

            if st.button("Logout"):
                if not st.session_state.developer:
                    if not st.session_state.first_prompt:
                        profile_text=generate_user_profile_summary(profile_text, summarize_chat_history(st.session_state.chat_history)) or profile_text
                        save_user_profile(username, profile_text)
                st.session_state.authenticated = False
                st.session_state.developer=False
                st.session_state.user_info = None
                st.session_state.username=''
                st.rerun()

            if not st.session_state.developer:
                st.markdown("---")
                st.header("ℹ️ Enhanced Features")
                st.write("✅ **Personalized responses**")
                st.write("✅ **Real-time sentiment analysis**")
                st.write("✅ **Crisis risk assessment**")
                st.write("✅ **Professional mental health support**")
                ##################### Make changes here ############
                st.markdown("---")
                st.header("🚨 Crisis Resources")
                st.write("**Emergency contacts:**")
                st.write("- **+233 244 846 701** (Mental Health Authority Helpline)")
                st.write("- **112** (Emergency Line)")
                st.write("- **+233 303 932 545** (Youth Helpline)")
                
            elif st.session_state.developer:
                if st.button("Configure"):
                    if st.button("Counselor Email"):
                        counselor_email=st.text_input("Type new email...")
                    if st.button("Agent Details"):
                            st.write("Provide the details")
                            smtp_email=st.text_input("Email..")
                            smtp_password=st.text_input("Password..", type="password")
                    if st.button("FEM Metric"):
                        st.session_state.fem_metric=True
                        st.write("FEM Metric Enabled")
        # Main chat interface
        st.subheader(f"💬 Chat with WellBot - Hello {user_info['name']}!")

        # Create columns for input and response
        facialExp=st.checkbox(label='Enable Facial Expression Awareness', value=False)
        if facialExp:
            detector=load_fem_model()
            input_col, response_col=st.columns([3,2])
        else:
            input_col, response_col=st.columns([1,1])
        with input_col:
            st.subheader("💭 Your Message")
            if facialExp:
                input_area, photo_area=st.columns([4,1], vertical_alignment="center")
                with input_area:
                    with st.form(key="chat_form", clear_on_submit=True):
                        input_text = st.text_area(
                            "Share what's on your mind:",
                            placeholder=f"Hi {user_info['name']}, I'm here to listen and provide support...",
                            height=90
                        )
                        send_clicked = st.form_submit_button("Send 📤", type="primary")
                with photo_area:
                    photo=st.camera_input("", key="facial input")
            else:
                with st.form(key="chat_form", clear_on_submit=True):
                    input_text = st.text_area(
                        "Share what's on your mind:",
                        placeholder=f"Hi {user_info['name']}, I'm here to listen and provide support...",
                        height=90
                    )
                    send_clicked = st.form_submit_button("Send 📤", type="primary")
                
            # Clear chat button
            if st.button("Clear Chat 🗑️"):
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                st.rerun()

            # Handle form submission
            if send_clicked and input_text.strip():
                with st.spinner("🤖 Processing..."):
                    # Initialize chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    # Process facial expression if enabled
                    if facialExp: 
                        gradient=get_fem(facialExp, detector, photo)
                        if st.session_state.fem_metric:
                            try:
                                with st.sidebar.status("Calibrating model..."):
                                    st.sidebar.metric("FEM",gradient)
                                    if st.sidebar.expander("Emotion Array"):
                                        try:
                                            st.sidebar.markdown(emotions)
                                        except:
                                            st.error("Emotion Array Malfunction")
                            except: st.error("Fem Metric Display Failed")
                    else: gradient=0.0
                    # Construct prompt with summary
                    if st.session_state.chat_history:
                        st.session_state.first_prompt=False
                    if st.session_state.first_prompt:
                        chat_prefix=profile_text
                        st.session_state.first_prompt=False
                    else: chat_prefix=summarize_chat_history(st.session_state.chat_history) or ""
                    summary_input=f"{chat_prefix}\n{input_text.strip()}"
                    # Analyze sentiment for the message
                    sentiment_score, risk_score, crisis_level, sentiment_label = analyze_sentiment_and_risk(input_text.strip())

                    # Add to chat history with sentiment data
                    st.session_state.chat_history.append({
                        'user': input_text.strip(),
                        'timestamp': time.time(),
                        'risk_score': risk_score,
                        'crisis_level': crisis_level,
                        'sentiment_label': sentiment_label,
                        'sentiment_score': sentiment_score
                    })
            elif send_clicked:
                st.warning("Please enter a message first.")
        with response_col:
            st.subheader("🤖 Personalized Responses")
            # Display current conversation with analysis
            if send_clicked and input_text.strip():
                ndisp=False
                st.session_state.chat_history[-1]['bot']=generate_response(summary_input, gradient, user_info['email'], user_info['name'], user_info.get('phone', 'Not provided'))
            elif 'chat_history' in st.session_state and st.session_state.chat_history:
                    # Only access the last element if the list is non-empty
                    if st.session_state.chat_history[-1].get('bot') is not None:
                        ndisp=False
                        with st.container(border=True):
                            st.write(f"### 🤖 WellBot (for {username}):")
                            st.write(st.session_state.chat_history[-1]['bot'])
            else:
                ndisp=True
                st.info(f"💡 Your personalized conversations will appear here, {user_info['name']}...")
                st.write(f"**Tell me what's on your mind {username}**")  

   
        # Chat history with sentiment analysis
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.markdown("#### 💬 Recent Conversations")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                # Get sentiment data if available
                risk_score = chat.get('risk_score', 0)
                crisis_level = chat.get('crisis_level', 'LOW')
                sentiment_label = chat.get('sentiment_label', 'NEUTRAL')
                sentiment_score = chat.get('sentiment_score', 0)

                # Color code the expander based on risk level
                if risk_score >= 6:
                    expander_label = f"🚨 Conversation {len(st.session_state.chat_history) - i} - {crisis_level} RISK ({risk_score}/10)"
                elif risk_score >= 4:
                    expander_label = f"⚠️ Conversation {len(st.session_state.chat_history) - i} - {crisis_level} RISK ({risk_score}/10)"
                else:
                    expander_label = f"💬 Conversation {len(st.session_state.chat_history) - i} - {crisis_level} ({risk_score}/10)"

                with st.expander(expander_label, expanded=False):
                    st.markdown("**You:**")
                    st.info(chat['user'])
                    # Display sentiment metrics for this message
                    with st.container(border=True):
                            st.write(f" 🤖 WellBot:")
                            st.write(chat['bot'])
                    # '''if 'risk_score' in chat:
                    #     col1, col2, col3 = st.columns(3)
                    #     with col1:
                    #         st.metric("Risk", f"{risk_score}/10")
                    #     with col2:
                    #         st.metric("Level", crisis_level)
                    #     with col3:
                    #         sentiment_color = "🟢" if sentiment_score > 0 else "🔴" if sentiment_score < -0.2 else "🟡"
                    #         st.metric("Sentiment", f"{sentiment_color} {sentiment_label}")'''

                    st.caption(f"⏰ {time.strftime('%H:%M:%S', time.localtime(chat['timestamp']))}")
        else:
            if not ndisp:
                st.info(f"💡 Your personalized conversations will appear here, {user_info['name']}...")
            st.markdown("**Features:**")
            st.write("• Personalized responses using your name")
            st.write("• Real-time sentiment analysis")
            st.write("• Professional mental health resources")
            st.write("• Secure authentication")

        # Quick test buttons
        st.markdown("---")
        st.subheader("🧪 Quick Scenarios")

        # Test buttons
        col1, col2, col3 = st.columns(3)

        with col2:
            if st.button("😊 I'm feeling great today!"):
                test_input = "I'm feeling great today!"
                st.info(f"**Test Input**: {test_input}")
                generate_response(test_input, 0.0,user_info['email'], user_info['name'], user_info.get('phone', 'Not provided'))

        with col1:
            if st.button("😔 I feel depressed"):
                test_input = "I feel depressed."
                st.info(f"**Test Input**: {test_input}")
                generate_response(test_input, 0.0,user_info['email'], user_info['name'], user_info.get('phone', 'Not provided'))

        with col3:
            if st.button("😔 I'm struggling to sleep"):
                test_input = "I'm struggling to sleep"
                st.info(f"**Test Input**: {test_input}")
                generate_response(test_input, 0.0,user_info['email'], user_info['name'], user_info.get('phone', 'Not provided'))
        if st.session_state.developer:
            # Sentiment analysis tester
            st.markdown("---")
            st.subheader("🔍 Sentiment Analysis Tester")
            with st.form("sentiment_test_form"):
                test_message = st.text_area("Enter a message to test sentiment analysis:",
                                        placeholder="Type any message to see how it's analyzed...")
                test_button = st.form_submit_button("Analyze Sentiment")

                if test_button and test_message.strip():
                    sentiment_score, risk_score, crisis_level, sentiment_label = analyze_sentiment_and_risk(test_message)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Risk Score", f"{risk_score}/10")
                    with col2:
                        st.metric("Crisis Level", crisis_level)
                    with col3:
                        st.metric("Sentiment", sentiment_label)
                    with col4:
                        sentiment_color = "🟢" if sentiment_score > 0 else "🔴" if sentiment_score < -0.2 else "🟡"
                        st.metric("Mood", f"{sentiment_color} {sentiment_score:.2f}")

                    # Show what would happen
                    if risk_score >= 6:
                        st.error("🚨 **HIGH RISK**: Crisis intervention would be triggered + Email sent")
                    elif risk_score >= 4:
                        st.warning("⚠️ **MODERATE RISK**: Automatic email alert would be sent")
                    elif risk_score >= 2:
                        st.info("💙 **LOW-MODERATE**: Additional resources would be provided")
                    else:
                        st.success("✅ **LOW RISK**: Normal supportive response")
if __name__ == '__main__':
    ndisp=True
    smtp_email=''
    smtp_password=''
    counselor_email=''
    emotions=None
    main()
