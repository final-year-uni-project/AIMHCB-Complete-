import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(r"C:\Users\Nazir Alabi\AIMHCB-Complete-\.env")

def initialize(value: int):
    '''Initialize relevant session variables to page default values'''
    try:
        # Login page defaults
        if value==0:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py location
            USER_DB_FILE = os.path.join(BASE_DIR, "data", "users.txt")
            PROFILE_DIR = os.path.join(BASE_DIR, "data", "profiles")

            if 'authenticated' not in st.session_state:
                st.session_state.authenticated=False
            
            if 'role' not in st.session_state:
                st.session_state.role=0
            
            if 'lock' not in st.session_state:
                st.session_state.lock = False
            
            if 'system_variables' not in st.session_state:
                st.session_state.system_variables={
                    'user_db_file': USER_DB_FILE,
                    'profile_dir': PROFILE_DIR,
                    'resource_1': None,
                    'resource_2': None,
                    'groq_client': Groq(api_key=os.getenv("API_KEY")),
                    'model': "llama-3.3-70b-versatile",
                    'detector': None,
                    'crisis_keywords': [],
                    'counselor_credentials':{},
                    'counselor_email':"aerickegreene12@gmail.com",
                    'smtp_email': "",
                    'smtp_password':""
                }

            if 'username' not in st.session_state:
                st.session_state.username = ""
            
            if 'user_info' not in st.session_state:
                st.session_state.user_info = {}
            
            if 'chat_info' not in st.session_state:
                st.session_state.chat_info={
                    'first_prompt':True,
                    'fem_feature':False,
                    'fem_acknowledged':True,
                    'pp':"",
                    'chat_summary':""
                }

            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
        
        # User page defaults
        if value==1:
            if 'authenticated' not in st.session_state:
                st.session_state.authenticated = True
            
            if 'role' not in st.session_state:
                st.session_state.role=1
            
            if 'lock' not in st.session_state:
                st.session_state.lock = False
            
            if 'user_info' not in st.session_state:
                st.session_state.user_info = {'name': 'Demo User', 'email': '', 'phone': 'Not provided'}
            
            if 'username' not in st.session_state:
                st.session_state.username = ""
            
            if 'system_variables' not in st.session_state:
                st.session_state.system_variables={
                    'user_db_file': r"C:\Users\Nazir Alabi\AIMHCB-Complete-\data\users.txt",
                    'profile_dir': r"C:\Users\Nazir Alabi\AIMHCB-Complete-\data\profiles",
                    'resource_1': None,
                    'resource_2': None,
                    'groq_client': Groq(api_key=os.getenv("API_KEY")),
                    'model': "llama-3.3-70b-versatile",
                    'detector': None,
                    'crisis_keywords': [],
                    'counselor_credentials':{}
                }
            
            if 'chat_info' not in st.session_state:
                st.session_state.chat_info={
                    'first_prompt':True,
                    'fem_feature':False,
                    'fem_acknowledged':True,
                    'pp':"",
                    'chat_summary':""
                }
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            if 'role' not in st.session_state:
                st.session_state.role=None

            
    except Exception as e:
        st.error(f"Initialization error:\n {e}")

                