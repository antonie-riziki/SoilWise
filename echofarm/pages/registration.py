import streamlit as st
import pandas as pd
import africastalking
import os
import requests
import google.generativeai as genai

from streamlit_lottie import st_lottie


from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime

col1, col2 = st.columns(2)




with col1:
	with st.form(key="user_registration"):
	    st.subheader("User Self Registration")
	    names = st.text_input("Official Names")
	    gender_text = st.write('Gender')
	    
	    chk_male, chk_female = st.columns(2)
	    
	    with chk_male:
	    	gender = st.checkbox('Male')
	    
	    with chk_female: 
	    	gender = st.checkbox('Female')
	    
	    username = st.text_input('Username:')
	    email = st.text_input("Email: ")
	    phone_number = st.number_input("Phone Number:", value=None, min_value=0, max_value=int(10e10))
	    password = st.text_input('Passowrd', type="password")
	    confirm_password = st.text_input('Confirm password', type='password')

	    checkbox_val = st.checkbox("Subscribe to our Newsletter")

	    submit_personal_details = st.form_submit_button("Submit")

	    # Every form must have a submit button.
	    if password != confirm_password:
	    	st.error('Password mismatch', icon='⚠️')

	    else:
		    
		    if not (email and password):
		    	st.warning('Please enter your credentials!', icon='⚠️')
		    else:
		    	st.success('Proceed to engaging with the system!', icon='👉')

		    	

		    	if submit_personal_details:

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

	

	# st.write("Outside the form")


def load_lottieurl(url: str):
	r = requests.get(url)
	if r.status_code != 200:
		return None
	else:
		return r.json()


with col2:
	reg_lottie = load_lottieurl("https://lottie.host/701a9d68-8f75-41a1-8c96-3e4b026a3d3f/zeKp8UyfVz.json")
	st_lottie(reg_lottie)
	st.image('https://www.google.com/imgres?q=soil%20crop%20yield%20africa%20women&imgurl=https%3A%2F%2Fcega.berkeley.edu%2Fwp-content%2Fuploads%2F2024%2F01%2FAdobeStock_525575233-scaled-1.jpeg&imgrefurl=https%3A%2F%2Fcega.berkeley.edu%2Fcollection%2Fthe-agribroom-app-targeted-bundled-digital-agricultural-technology-for-inclusive-agricultural-productivity%2F&docid=qSoSERCEBSivlM&tbnid=XJKebINm0_2TZM&vet=12ahUKEwjBp4aR_uOLAxXx_7sIHWkrHtY4ChAzegQIShAA..i&w=1200&h=800&hcb=2&itg=1&ved=2ahUKEwjBp4aR_uOLAxXx_7sIHWkrHtY4ChAzegQIShAA', width=700)
	# st.image('./assets/img/registration2.jpeg', width=700)
