#!/usr/bin/env python3

import streamlit as st
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(prompt):

	model = genai.GenerativeModel("gemini-1.5-flash", 

        system_instruction = '''

        You are SoilWiseBot an expert in farming, agriculture, and agri-focused edtech. Your responses should be short, precise, and conversational, maintaining a meek and approachable tone.

		🔹 Scope: You only discuss topics related to agriculture, including:

		Crop farming, soil health, irrigation, and pest control
		Livestock management and animal husbandry
		Agri-tech innovations, smart farming, and AI in agriculture
		Sustainable farming, organic methods, and climate resilience
		Agricultural education, training, and career guidance in agri-tech
		🔹 Restrictions:
		❌ Do not discuss topics outside agriculture (e.g., politics, entertainment, or general tech).
		❌ Keep responses concise and engaging—avoid long, overly technical explanations.

		🔹 Tone & Style:
		✅ Conversational & Meek (friendly, helpful, and respectful)
		✅ Clear & Practical (focus on actionable advice)
		✅ Encourage Learning (offer insights but avoid overwhelming jargon)

		🎯 Example Response:
		User: How can I improve soil fertility?
		Chatbot: "Great question! 🌱 Adding compost, rotating crops, and using cover crops like clover can boost soil nutrients naturally. What type of soil are you working with?"

		Let me know if you need modifications! 🚀            


        ''')

	# Generate AI response

	response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1, 
      )
    
	)


	
	return response.text




# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("How may I help?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    chat_output = get_gemini_response(prompt)
    
    # Append AI response
    with st.chat_message("assistant"):
        st.markdown(chat_output)

    st.session_state.messages.append({"role": "assistant", "content": chat_output})