import sys
import os
import streamlit as st

# --- Ensure project root is in Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Now safe to import echofarm modules ---
from echofarm.mcp.client import MCPClient, MCPClientError
from echofarm.mcp.config import MCP_SERVER_URL

# st.set_page_config(page_title="SoilWise Chatbot + MCP", layout="wide")

# st.title("SoilWise — Chatbot (MCP client)")

# initialize client
client = MCPClient(MCP_SERVER_URL)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**SoilWise:** {msg['text']}")

    user_input = st.text_input("Send a message", key="user_input")
    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            # simple keyword trigger: if user asks about recommendation, call MCP
            lowered = user_input.lower()
            if "recommend" in lowered or "what to plant" in lowered or "crop" in lowered:
                # open a dialog in the right pane to enter sensor data (or default)
                st.session_state.chat_history.append({"role": "assistant", "text": "Please provide soil metrics on the right pane (or use defaults), then click 'Get Recommendation'."})
            else:
                # fallback quick reply
                st.session_state.chat_history.append({"role": "assistant", "text": "I can recommend crops based on soil metrics. Ask me to recommend crops or click 'Tools'."})
        st.rerun()

with col2:
    st.header("Tools / MCP")
    try:
        tools = client.list_tools()
        st.write("Discovered tools:")
        st.json(tools)
    except Exception as e:
        st.error(f"Could not reach MCP server at {MCP_SERVER_URL}: {e}")
        st.stop()

    st.subheader("Invoke soil_recommendation")
    with st.form("invoke_form"):
        ph = st.slider("pH", 3.0, 9.0, 6.5)
        moisture = st.slider("Moisture (%)", 0, 100, 40)
        nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0.0, max_value=1000.0, value=60.0)
        phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0.0, max_value=1000.0, value=40.0)
        potassium = st.number_input("Potassium (mg/kg)", min_value=0.0, max_value=1000.0, value=30.0)
        temperature = st.number_input("Soil Temp (°C)", value=25.0)
        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        payload = {
            "ph": ph, "moisture": moisture, "nitrogen": nitrogen,
            "phosphorus": phosphorus, "potassium": potassium, "temperature": temperature
        }
        try:
            result = client.invoke_tool("soil_recommendation", payload)
            # display and add to chat history
            pretty = result.get("result", result)
            st.json(pretty)
            st.session_state.chat_history.append({"role": "assistant", "text": f"Recommendation: {pretty}"})
            st.rerun()
        except MCPClientError as me:
            st.error(f"Invoke failed: {me}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
