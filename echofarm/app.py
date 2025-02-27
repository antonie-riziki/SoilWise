import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
import plotly.express as px
import os
import csv
import sys


registration_page = st.Page("./pages/registration.py", title="Signup", icon=":material/app_registration:")
home_page = st.Page("./pages/main.py", title="Research", icon=":material/analytics:")
agri_shield_page = st.Page("./pages/agri_shield.py", title="agri shield", icon=":material/grass:")
irrigate_page = st.Page("./pages/irrigAIte.py", title="irrigAIte", icon=":material/agriculture:")
soilwise_page = st.Page("./pages/soilwise.py", title="Soilwise", icon=":material/yard:")
chatbot = st.Page("./pages/chatbot.py", title="ChatBot", icon=":material/chat:")



pg = st.navigation([registration_page, home_page, soilwise_page, irrigate_page, agri_shield_page, chatbot])


st.set_page_config(
    page_title="EchoFarm",
    page_icon="🌱🪴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.echominds.africa',
        'Report a bug': "https://www.echominds.africa",
        'About': "# We are a leading insights and predicting big data application, Try *EchoFarm* and experience reality!"
    }
)

pg.run()
