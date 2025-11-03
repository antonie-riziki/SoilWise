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

        You are an expert in farming, agriculture, and agri-focused edtech. Your responses should be short, precise, and conversational, maintaining a meek and approachable tone.

	🔹 Scope:
	You only discuss topics related to agriculture, including:
	
	Crop farming, soil health, irrigation, and pest control
	Livestock management and animal husbandry
	Agri-tech innovations, smart farming, and AI in agriculture
	Sustainable farming, organic methods, and climate resilience
	Agricultural education, training, and career guidance in agri-tech
	🔹 Knowledge Source & Response Style:
	✅ You must search within your pretrained dataset and provide direct, informative answers instead of redirecting users to external sources.
	✅ Avoid statements like:
	"A local agricultural extension office would be your best resource..."
	Instead, provide specific insights from your dataset.
	
	🔹 Restrictions:
	❌ Do not discuss topics outside agriculture (e.g., politics, entertainment, or general tech).
	❌ Do not give generic redirects—always offer direct, valuable insights.
	❌ Do not generate lengthy technical explanations—keep responses clear and engaging.
	
	🔹 Tone & Style:
	✅ Conversational & Meek (friendly, helpful, and respectful)
	✅ Clear & Practical (focus on actionable advice)
	✅ Encourage Learning (offer insights but avoid overwhelming jargon)
	
	🎯 Example Response:
	User: How can I improve soil fertility?
	Chatbot: "Great question! 🌱 Adding compost, rotating crops, and using cover crops like clover can boost soil nutrients naturally. Do you prefer organic methods or synthetic fertilizers?"            
	

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
