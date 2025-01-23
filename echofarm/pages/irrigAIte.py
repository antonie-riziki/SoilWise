import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sb 
import numpy as np 
import csv
import cv2
import warnings

header = st.container()
home_sidebar = st.container()

with header:
	st.title("IrrigAIte")

	st.markdown('Smart Watering for Smarter Yields')

	st.write('''
		IrrigAIte is the future of precision farming. Leveraging IoT technology and real-time data monitoring, this automated irrigation 
		system measures soil moisture levels and activates watering only when needed. Designed to save water, reduce costs, and boost crop 
		health, IrrigAIte ensures every drop counts. With its AI-driven efficiency, you can focus on farming while it takes care of your 
		crops hydration needs.

		''')






	
