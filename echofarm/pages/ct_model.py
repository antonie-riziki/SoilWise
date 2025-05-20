import pandas as pd 
import cv2
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sb 
import numpy as np 
import csv
import os
import sys
import tempfile
import warnings
import joblib
import google.generativeai as genai



from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, r2_score, mean_absolute_error, mean_squared_error


from dotenv import load_dotenv

sys.path.insert(1, './pages')

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

le = LabelEncoder()

rfr = RandomForestRegressor()
rfc = RandomForestClassifier()


def prepare_train_test_for_soil_quality(df):
	x = df.drop(columns=['Date', 'Crop_Type', 'Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'Soil_Quality'])
	y = df['Soil_Quality']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

	rfr.fit(x_train, y_train)

	soil_quality_model = joblib.dump(value=rfr, filename='.\model\sqm.pkl')

	return x_train, x_test, y_train, y_test 


def prepare_train_test_for_soil_ph(df):
	x = df.drop(columns=['Date', 'Crop_Type', 'Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed'])
	y = df['Soil_pH']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

	rfr.fit(x_train, y_train)

	soil_ph_model = joblib.dump(value=rfr, filename='.\model\sph.pkl')

	return x_train, x_test, y_train, y_test


def prepare_train_test_for_soil_type(df):
	x = df.drop(columns=['Date', 'Crop_Type', 'Temperature', 'Humidity', 'Wind_Speed'])
	x = x[['N', 'P', 'K', 'Crop_Yield', 'Soil_Quality', 'Soil_pH']]
	y = df['Soil_Type']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

	rfc.fit(x_train, y_train)

	soil_type_model = joblib.dump(value=rfc, filename='.\model\styp.pkl')

	return x_train, x_test, y_train, y_test


def prepare_train_test_for_crop_type(df):
	x = df.drop(columns=['Date', 'Crop_Type', 'Temperature', 'Humidity', 'Wind_Speed'])
	x = x[['N', 'P', 'K', 'Crop_Yield', 'Soil_Quality', 'Soil_pH', 'Soil_Type']]
	y = df['Crop_Type']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

	rfc.fit(x_train, y_train)

	crop_type_model = joblib.dump(value=rfc, filename='.\model\ctyp.pkl')

	return x_train, x_test, y_train, y_test


# def gemini_chatbot():
# 	model = genai.GenerativeModel("gemini-1.5-flash")
# 	# response = model.generate_content("Write a story about a magic backpack.")
# 	chat = model.start_chat(history=[])
# 	# return chat
# 	# print(response.text)


def get_gemini_response(prompt, question):

	model = genai.GenerativeModel("gemini-2.0-flash")
	# response = model.generate_content("Write a story about a magic backpack.")
	chat = model.start_chat(history=[])
	# return chat
	# print(response.text)


	# chat = gemini_chatbot()
	response=chat.send_message((
		prompt + question).replace('\n', ' ').replace('\r', ''), stream=True)
	return st.write(response)


	# st.set_page_config(page_title='Mental Health Chatbot')

	st.header('GEMINI LLM GENERATIVE AI')



	# Initialize session state - saves the history of the chat
	if 'chat_history' not in st.session_state:
		st.session_state['chat_history'] = []


	input = st.text_input("Input: ", key="input")
	submit = st.button('submit: ')

	if submit and input:
		response = get_gemini_response(input)

		st.session_state['chat_history'].append(('You: ', input))
		st.subheader('the response is:')

		for chunk in response:
			st.write(chunk.text)
			st.session_state['chat_history'].append(('Chatbot', chunk.text))


	return chunk.text


def get_recommended_crop(data):
	current_dir = os.path.dirname(os.path.abspath(__file__))

	# Construct the path to the image
	csv_path = os.path.join(current_dir, '../src/crop_yield_dataset.csv')
	
	new_df = pd.read_csv(csv_path)

	x = new_df[['N', 'P', 'K', 'Crop_Yield', 'Soil_Quality', 'Soil_pH', 'Soil_Type']]
	y = new_df['Crop_Type']

	# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	x['Soil_Type'] = le.fit_transform(x['Soil_Type'])
	y_encoded = le.fit_transform(y)

	scaler = StandardScaler()

	x_scaled = scaler.fit_transform(x)


	logr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	logr_model.fit(x_scaled, y_encoded)


	new_data = np.array([data])

	# Scale new data
	new_data_scaled = scaler.transform(new_data)

	# Predict probabilities
	probabilities = logr_model.predict_proba(new_data_scaled)

	# Get the recommended crop
	recommended_index = np.argmax(probabilities)
	recommended_crop = le.inverse_transform([recommended_index])[0]

	st.subheader(f''' We also Recommend: {recommended_crop}''')
	st.write(f'''Based on the current soil conditions, {recommended_crop} is a highly suitable 
		choice for cultivation. Monitoring soil health and maintaining optimal conditions can further enhance productivity ''')

	st.subheader(f''' Intercropping associated with {recommended_crop}''')

	prompt = (f''' Suggest two or more crops that can be grown together with {recommended_crop} in an intercropping system in the Kenyan ecosystem, considering nutrient needs and benefits for each crop. 
		For example, beans release nitrogen, which benefits nitrogen-demanding crops like maize. Provide examples for different soil types or 
		climates if possible. Just list the crops dont provide too much information  ''')

	model = genai.GenerativeModel("gemini-1.5-flash", 
		# system_instruction = "You are an expert agricultural assistant named SoilWise. Your purpose is to provide farmers with accurate, practical, and localized advice on soil quality, crop recommendations, farming techniques, and sustainable agricultural practices. Respond in a friendly and professional tone, ensuring your guidance is easy to understand and actionable."
)
	response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)

	st.write(response.text)


	st.subheader(f''' Crop Rotation associated with {recommended_crop}''')

	prompt = (f'''In summary points List crop rotation based on {recommended_crop} for sustainable soil health and nutrient management in Kenyan ecosystem and have high-yield crops for economic benefit. 
		Provide recommendations tailored to {recommended_crop} just for year 1. Just list the crops dont provide too much information''')

	model = genai.GenerativeModel("gemini-1.5-flash", 
		# system_instruction = "You are an expert agricultural assistant named SoilWise. Your purpose is to provide farmers with accurate, practical, and localized advice on soil quality, crop recommendations, farming techniques, and sustainable agricultural practices. Respond in a friendly and professional tone, ensuring your guidance is easy to understand and actionable."
)
	response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)

	st.write(response.text)

	return recommended_crop
	


def get_ai_content(prompt):

	model = genai.GenerativeModel("gemini-2.0-flash", 
		system_instruction = '''
  You are an expert agricultural assistant named SoilWise. Your purpose is to provide precise, practical, and localized advice on soil quality, crop recommendations, farming techniques, and sustainable agriculture. 
  Keep responses short, clear, and data-driven, avoiding excessive details. Prioritize quantifiable insights (e.g., optimal soil pH, recommended fertilizer ratios, expected yield per acre) to ensure farmers can take immediate, informed action. 
  Maintain a meek and professional tone while making information easy to understand and apply
  '''
)
	response = model.generate_content(
    prompt + ' in Kenya',
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
	
	st.write(response.text)

	# st.write(response.text)


def the_explainer(prompt):

	model = genai.GenerativeModel("gemini-1.5-flash", 
		system_instruction = '''
					You are an intelligent data analysis assistant designed to help users understand insights derived from grouped farmers datasets. 
					Your primary objective is to provide clear, concise, and engaging explanations of visualized data based on the user's selected country or region, the specific series being analyzed, and key insights.

					Your responsibilities include:
					1. Explaining the purpose of the graph and its relevance to the selected parameters.
					2. Highlighting key insights in a structured and easy-to-understand manner.
					3. Encouraging users to interpret trends, disparities, or patterns observed in the graph.
					4. Using a professional yet approachable tone to ensure the explanation is interactive and user-friendly.

					Make sure your explanations are tailored to the user's selections and provide actionable insights wherever applicable also note that the dataset is strictly based on farmers data collection also summarize and quantify the results and possible as you can.
					''')

	response = model.generate_content(
    prompt,
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
	
	st.write(response.text)



def get_crop_summary(prompt):

	model = genai.GenerativeModel("gemini-2.0-flash", 
		system_instruction = '''
  You are an expert agricultural assistant named SoilWise. Your purpose is to provide farmers with accurate, practical, and localized advice on soil quality, crop recommendations, farming techniques, and sustainable agricultural practices. Respond in a friendly and professional tone, ensuring your guidance is easy to understand and actionable as well as quantifying your reponse as much as possible.'''
)
	response = model.generate_content(
    prompt + ' in Kenya',
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)

	st.write(response.text)








