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
import random
import africastalking

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, r2_score, mean_absolute_error, mean_squared_error

from PIL import Image


sys.path.insert(1, './pages')
print(sys.path.insert(1, '../pages/'))

from ct_model import  prepare_train_test_for_soil_quality, prepare_train_test_for_soil_ph, prepare_train_test_for_soil_type, prepare_train_test_for_crop_type, get_recommended_crop, get_ai_content, the_explainer, get_crop_summary

from dotenv import load_dotenv

load_dotenv()

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 1000)

sb.set_style("darkgrid")

warnings.filterwarnings("ignore")

header = st.container()
home_sidebar = st.container()
main_body = st.container()


with header:
	st.title("SOILWISE: PROOF OF CONCEPT")

	st.markdown('Discover the Secrets Beneath Your Feet')

	st.write('''
		SoilWise is your intelligent companion for understanding and optimizing soil health. Powered by advanced IoT sensors and 
		Machine Learning algorithms, SoilWise analyzes your soil's type, moisture levels, pH, and nutrient content. It then provides 
		tailored recommendations for the best crops to plant, ensuring maximum yield and sustainability. Whether you're a seasoned farmer 
		or a first-time grower, SoilWise takes the guesswork out of farming and puts science in your hands.

		''')
	
	# tab1,tab2 = st.tabs(['Capture Image', 'Take a Video'])

with main_body:
	
	df = pd.read_csv(r'./src/crop_yield_dataset.csv')

	le = LabelEncoder()
	def get_categorical_columns(df):
	    category = []
	    for i in df.select_dtypes(include=["object"]):
	        df[i] = le.fit_transform(df[i])
	    plt.figure(figsize=(12, 8))
	    return sb.heatmap(df.corr(), annot=True, linewidths=0.5, cmap="viridis")

	get_categorical_columns(df)


	##############################################################################################################################


	# SOIL QUALITY MODEL


	###############################################################################################################################

	# x_train, x_test, y_train, y_test = prepare_train_test_for_soil_quality(df)

	def get_soil_quality(data):

	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    load_sqm = joblib.load('./model/sqm.pkl')

	    pred = load_sqm.predict(input_data_array_reshaped)
	    
	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    pred.flatten().tolist()
	    data.append(float(pred[0]))  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]

	    st.write("Soil Quality")

	    if pred >= 70:
	        st.write(f'''The soil quality score is {round(pred[0])}, which indicates excellent fertility and suitability for most crops. 
	                This soil can support high-yield crops with minimal intervention. To maintain this quality, 
	                consider crop rotation and adding organic matter occasionally to replenish nutrients.''')
	    elif 50 <= pred < 70:
	        st.write(f'''The soil quality score is {round(pred[0])}, which reflects good fertility. 
	                While the soil is suitable for many crops, applying fertilizers or compost can help optimize yields. 
	                Regular soil testing is recommended to ensure continued fertility.''')
	    elif 30 <= pred < 50:
	        st.write(f'''The soil quality score is {round(pred[0])}, indicating moderate fertility. 
	                The soil may require improvements such as adding organic matter, fertilizers, or addressing drainage issues. 
	                Growing less demanding crops or improving soil structure with organic amendments can boost productivity.''')
	    elif 15 <= pred < 30:
	        st.write(f'''The soil quality score is {round(pred[0])}, suggesting low fertility. 
	                This soil is not ideal for most crops without significant amendments. Consider applying a combination of 
	                organic matter, fertilizers, and soil conditioners to improve its fertility. Planting cover crops may help.''')
	    else:
	        st.write(f'''The soil quality score is {round(pred[0])}, which indicates very poor fertility. 
	                Extensive soil management is required to grow crops. Focus on improving soil structure, increasing organic matter, 
	                and addressing potential issues like salinity or compaction before planting.''')
	    st.write("")

	    return pred, data


	##############################################################################################################################


	# SOIL PH MODEL


	############################################################################################################################### 

	# x_train_2, x_test_2, y_train_2, y_test_2 = prepare_train_test_for_soil_ph(df)

	def get_soil_ph(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    load_sph = joblib.load('./model/sph.pkl')

	    pred = load_sph.predict(input_data_array_reshaped)

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    pred.flatten().tolist()
	    data.append(float(pred[0]))  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    
	    st.write("Soil PH")
	    if 6.2 <= pred <= 6.8:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is within the optimal range for most crops (6.2 - 6.8). "
	                "This means your soil is slightly acidic, making it ideal for nutrient absorption. "
	                "To maintain or improve this balance, consider regular pH testing and avoid overusing acidic or alkaline fertilizers.")
	    elif pred < 6.2:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is lower than the optimal range (6.2 - 6.8). "
	                "This indicates your soil is more acidic, which can limit nutrient availability. "
	                "Consider applying lime to raise the pH and improve soil conditions.")
	    else:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is higher than the optimal range (6.2 - 6.8). "
	                "This indicates alkaline soil, which might reduce nutrient uptake. "
	                "Adding sulfur or organic matter can help lower the pH.")
	    st.write("")
	    return pred, data


	##############################################################################################################################

	
	# SOIL TYPE MODEL
	

	############################################################################################################################### 

	# x_train_3, x_test_3, y_train_3, y_test_3 = prepare_train_test_for_soil_type(df)

	def get_soil_type(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    load_styp = joblib.load('./model/styp.pkl')

	    pred = load_styp.predict(input_data_array_reshaped)

	    new_pred_val = pred[0]

	    st.write("Soil Type")

	    try:
	        if new_pred_val == 0:
	  
	            st.write(f'''The soil type is identified as Clay Soil. Clay soils are nutrient-rich and retain water well, 
				            but they can become compacted and drain poorly if not managed properly. To improve conditions, 
				            consider adding organic matter like compost or aged manure to enhance soil structure and aeration. 
				            Growing crops like rice, legumes, and sunflowers can be beneficial.''')

	        elif new_pred_val == 1:
	            st.write(f''''The soil type is identified as Loam Soil. Loam is the ideal soil type for most crops because it provides 
				            excellent drainage while retaining adequate moisture and nutrients. To maintain this balance, 
				            regularly add organic matter and avoid over-tilling. Crops such as vegetables, fruits, and grains thrive in loam soil.''')
	        
	        elif new_pred_val == 2:
	            st.write(f'''The soil type is identified as Peaty Soil. Peaty soils are high in organic matter and retain moisture well, 
				            but they can be acidic, which may limit nutrient availability. To optimize crop growth, consider adding lime to 
				            reduce acidity and improve pH balance. Suitable crops include root vegetables, berries, and shrubs. ''')

	        elif new_pred_val == 3:
	            st.write(f''' The soil type is identified as Saline Soil. Saline soils have high salt content, which can hinder plant growth. 
				            To improve conditions, focus on leaching salts by providing adequate drainage and using freshwater irrigation. 
				            Salt-tolerant crops such as barley, sugar beet, and certain grasses are recommended for this soil type. ''')
	        
	        elif new_pred_val == 4:
	            st.write(f''' The soil type is identified as Sand Soil. Sandy soils drain quickly and are easy to work with, but they retain 
				            fewer nutrients and moisture, making them less fertile. To enhance fertility, add organic matter like compost or 
				            mulch. Crops such as carrots, potatoes, and watermelon grow well in sandy soils. ''')
	        else:
	            st.write(f"Value out of range {pred}")
	    except ValueError as e:
	        st.write(f"Unexpected Error Occurred: {e}")

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    # pred.flatten().tolist()
	    data.append(new_pred_val)  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    st.write("")
	    return pred, data


	##############################################################################################################################

	
	# CROP TYPE MODEL
	

	############################################################################################################################### 

	# x_train_4, x_test_4, y_train_4, y_test_4 = prepare_train_test_for_crop_type(df)

	def get_crop_type(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    load_ctyp = joblib.load('./model/ctyp.pkl')

	    pred = load_ctyp.predict(input_data_array_reshaped)

	    new_pred_val = pred[0]

	    st.write("Crop Type Prediction")
	    
	    try:
	        if new_pred_val == 0:
	            st.write(f''' The predicted crop type is Barley. Barley grows best in well-drained, loam or sandy-loam soils with a pH range of 6.0 to 7.5.
                   			It thrives in moderately fertile soils with good moisture retention. To ensure optimal growth, avoid waterlogging, and apply nitrogen fertilizers during early growth stages. ''')

	        elif new_pred_val == 1:
	            st.write(f'''The predicted crop type is Maize. Maize grows well in loam or sandy-loam soils with good drainage and a pH range of 5.5 to 7.0. 
                  			This crop needs adequate nitrogen and potassium for optimal growth. Regular irrigation and weed control are crucial to ensure a healthy yield. ''')
	        
	        elif new_pred_val == 2:
	            st.write(f''' The predicted crop type is Cotton. Cotton prefers well-drained, loam or sandy-loam soils with a pH range of 5.8 to 7.0.
                   			This crop requires good fertility and warm temperatures for growth. Regular irrigation and the application of potassium-rich fertilizers can improve fiber quality. ''')

	        elif new_pred_val == 3:
	            st.write(f''' The predicted crop type is Potato. Potatoes prefer well-drained, sandy-loam soils with a pH range of 5.0 to 6.5.
                   			This crop requires soils rich in organic matter and potassium. Avoid waterlogging by improving soil drainage, 
                   			and consider mulching to retain soil moisture.''')

	        elif new_pred_val == 4:
	            st.write(f''' The predicted crop type is Rice. Rice thrives in clay or silty soils that can hold water for extended periods. 
		                 It requires soils with a pH range of 5.5 to 6.5 and high organic matter. Ensure proper water management and nutrient application, 
		                 including nitrogen, phosphorus, and potassium, for a high yield. ''')

	        elif new_pred_val == 5:
	            st.write(f''' The predicted crop type is Soybean. Soybeans grow well in loam or clay-loam soils with a pH range of 6.0 to 7.0. 
                    		These crops require moderate fertility and good drainage. Incorporating rhizobium inoculants can enhance nitrogen fixation and improve yield. ''')
	        
	        elif new_pred_val == 6:
	            st.write(f'''The predicted crop type is Sugarcane. Sugarcane thrives in deep, well-drained, loam or clay-loam soils with a pH range of 6.0 to 7.5. 
                      		It requires high fertility and abundant water for optimal growth. Regular irrigation, proper weed control, and the application of nitrogen and phosphorus fertilizers 
                      		are crucial for high yields. ''')

	        elif new_pred_val == 7:
	            st.write(f'''The predicted crop type is Sunflower. Sunflowers are well-suited for a wide range of soils, 
                      including loam and clay soils, as long as they are well-drained. Sunflowers thrive in soils with 
                      a pH range of 6.0 to 7.5 and moderate to high fertility. To maximize yield, ensure adequate sunlight 
                      and apply nitrogen-based fertilizers during the early growth stages. ''')

	        elif new_pred_val == 8:
	            st.write(f''' The predicted crop type is Tomato. Tomatoes thrive in well-drained, fertile, sandy-loam soils with a pH range of 6.0 to 6.8. 
                   			They require sufficient sunlight, regular watering, and balanced fertilizers to support their growth. Adding organic mulch can help retain moisture. ''')

	        elif new_pred_val == 9:
	            st.write(f'''The predicted crop type is Wheat. Wheat requires well-drained, loam or clay-loam soils with a pH between 6.0 and 7.0. 
		                  It grows best in moderately fertile soils with good moisture retention. Ensure proper irrigation during critical growth stages 
		                  and apply phosphorus fertilizers to enhance grain yield. ''')

	        else:
	            st.write(f"Value out of range {pred}")
	    except ValueError as e:
	        st.write(f"Unexpected Error Occurred: {e}")

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    # pred.flatten().tolist()
	    data.append(new_pred_val)  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    
	    st.write("")
	    # st.write(data)
	    st.dataframe(pd.DataFrame([data], columns=['N', 'P', 'K', 'Crop_Yield', 'Soil_Quality', 'Soil_pH', 'Soil_Type', "Crop_Type"]))
	    return pred, data



	with st.form("user form fill"):
		col1, col2 = st.columns(2)

		with col1:
			nitrogen_values = st.number_input("Enter nitrogen level", df['N'].min(), df['N'].max(), (df['N'].mean()))
			phosphorus_values = st.number_input("Enter phosphorus level", df['P'].min(), df['P'].max(), (df['P'].mean()))

		with col2:
			potassium_values = st.number_input("Enter potassium level", df['K'].min(), df['K'].max(), (df['K'].mean()))
			crop_yield_values = st.number_input("Enter crop yield level", df['Crop_Yield'].min(), df['Crop_Yield'].max(), (df['Crop_Yield'].mean()))
		
		
		def get_crop_images(subpath):
			img_dir = "./assets/img/crop_type"  
			corn_folder = os.path.join(img_dir, subpath)
			return(corn_folder)

		def get_soil_images(subpath):
			soil_dir = "./assets/img/soil_type"
			soil_folder = os.path.join(soil_dir, subpath)
			return(soil_folder)


		if st.form_submit_button("Cultivate Wisdom"):
			col1, col2 = st.columns(2)

			with st.expander('Full Report', expanded=True):

				with col1:
					data = (nitrogen_values, phosphorus_values, potassium_values, crop_yield_values)
				
					soil_quality_prediction, soil_quality_data = get_soil_quality(data)

					soil_ph_prediction, soil_ph_data = get_soil_ph(soil_quality_data)

					soil_type_prediction, soil_type_data = get_soil_type(soil_ph_data)

					crop_type_prediction, crop_type_data = get_crop_type(soil_type_data)

					


				with col2:
					st.write("Visual Insights: Soil & Crop")
					# crop_type_prediction, crop_type_data = get_crop_type(soil_type_data)

					crop_dict = {
						'barley': 0,
						'corn': 1,
						'cotton': 2,
						'potato': 3,
						'rice':4,
						'soyabeans': 5,
						'sugarcane': 6,
						'sunflower': 7,
						'tomato': 8,
						'wheat': 9,
					}

					soil_dict = {
						'clay': 0,
						'loam': 1,
						'peaty': 2,
						'saline': 3,
						'sand': 4,
					}

					for val, key in enumerate(soil_dict):
						if soil_type_prediction == val:  # Check the prediction condition

							path = get_soil_images(key)
							
							if os.path.exists(path) and os.path.isdir(path):
								
								images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
								
								if images:
									random_image = random.choice(images)
									image_path = os.path.join(path, random_image)

									img = Image.open(image_path)
									st.image(img, caption=f"{key} soil", use_container_width=True)
								else:
									st.warning("No images found in the folder.")
							else:
								st.error(f"The folder does not exist")
								break
						else:
							pass

					for val, key in enumerate(crop_dict):
						if crop_type_prediction == val:  # Check the prediction condition

							path = get_crop_images(key)
							
							if os.path.exists(path) and os.path.isdir(path):
								
								images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
								
								if images:
									random_image = random.choice(images)
									image_path = os.path.join(path, random_image)

									img = Image.open(image_path)
									st.image(img, caption=f"{key}", use_container_width=True)
								else:
									st.warning("No images found in the folder.")
							else:
								st.error("The folder does not exist.")
						else:
							pass

			recommended_crop = get_recommended_crop(soil_type_data)


			for val, key in enumerate(crop_dict):
				if crop_type_prediction == val:
					tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Ploughing", "Sowing", "Adding Nutrients", "Irrigation", "Pests & Disease", "Harvest", "Storage", "Summary"])


					with tab1:
						st.title("Land Preparation")
						
						prompt = f''' What are the best practices for preparing the soil before planting {key}, 
						including ploughing techniques, depth, and timing? '''
						get_ai_content(prompt)

					with tab2:
						st.title("Seed Planting")
						
						prompt = f''' What is the recommended spacing, planting depth, and seed variety selection for 
						{key} sowing to ensure optimal growth? '''
						get_ai_content(prompt)
						

					with tab3:
						st.title("Nutrition Boosting")
						
						prompt = f''' What types of fertilizers or organic nutrients are recommended for {key} at different 
						growth stages, and how should they be applied? '''
						get_ai_content(prompt)

					with tab4:
						st.title("Land Watering")
						
						prompt = f''' What is the ideal irrigation schedule for {key} to ensure healthy growth without 
						overwatering or underwatering? '''
						get_ai_content(prompt)

					with tab5:
						st.title("Integrated Pest Management (IPM)")
						
						prompt = f''' What are the common pests and diseases that affect {key}, and how can they be effectively 
						controlled using sustainable methods? '''
						get_ai_content(prompt)

					with tab6:
						st.title("Abadunt Reaping")
						
						prompt = f''' When is the best time to harvest {key}, and what are the signs that indicate readiness for harvesting? '''
						get_ai_content(prompt)

					with tab7:
						st.title("Repository")
						
						prompt = f''' What are the best storage practices for {key} to maintain quality and prevent spoilage or pest infestation? '''
						get_ai_content(prompt)


					with tab8:
						st.title("Summary Report Details")
						
						prompt = f''' Generate a short planting report for {key} based on Kenyan ecosystem. Simplify the details into one line per point, 
								using quantified metrics where applicable. Ensure the report is concise, easy to understand, and suitable for sending via SMS. Example format:

									1. Crop: [Crop Type]
									2. Optimal Planting Time: [Time/Month related to Kenyan climatic conditions during each season]
									3. Required Fertilizer: [Amount and Type, also recommend best agro-input company in Kenya for fertilizer]
									4. Expected Yield: [Quantity in kg/acre for smallscale farmers in Kenya]
									5. Water Requirement: [Liters per acre/week or other relevant metrics]
									Provide additional points if relevant, while keeping each line brief and clear '''
						
						get_crop_summary(prompt)

	col1, col2, col3 = st.columns(3, gap="large", vertical_alignment="bottom")

	with col1:
		# get_sms =  st.button("Send to SMS", icon=":material/sms:")

		# if get_sms:

		with st.popover("Send to SMS", icon=":material/sms:"):

			with st.form(key="report"):
				phone_number = st.number_input('Phone Number', value=0, min_value=0, max_value=int(10e10))

				submit_report = st.form_submit_button("Send")

				def send_report():
					amount = "10"
					currency_code = "KES"


					recipients = [f"+254{str(phone_number)}"]
					# airtime_rec = "+254" + str(phone_number)
					print(recipients)
					print(phone_number)

					# Set your message
					message = f"Welcome to SoilWise! Revolutionizing farming with advanced soil testing for better yields & sustainable growth. Let's cultivate a greener future together!";
					# Set your shortCode or senderId
					sender = 20880
					try:
						# responses = airtime.send(phone_number=airtime_rec, amount=amount, currency_code=currency_code)
						response = sms.send(message, recipients, sender)
						
						print(response)
						# print(responses)
					except Exception as e:
						print(f'Houston, we have a problem: {e}')

			if submit_report not in st.session_state:
				send_report()
	
	with col2:
		with st.popover("Chat on Whatsapp", icon=":material/chat:"):

			st.info("Coming soon")

				

	with col3:
		voice_command = st.button("Sauti ya Shamba", icon=":material/mic:")

				


	# gemini_chatbot()
