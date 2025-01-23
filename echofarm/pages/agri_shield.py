import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sb 
import numpy as np  
import csv
import autoreload
import warnings
import cv2

header = st.container()
home_sidebar = st.container()

with header:
	st.title("AGRI SHIELD")
	st.markdown('Your Digital Protector Against Crop Threats')

	st.write('''
		Agri Shield is an AI-powered guardian for your farm. Using Computer Vision and Object Detection technologies, Agri Shield scans 
		your plants and produce to identify pests and diseases with pinpoint accuracy. It doesn’t stop there—Agri Shield provides 
		actionable insights and preventative measures to keep your crops safe from future infestations. With Agri Shield, you can protect 
		your harvest and secure your profits, one scan at a time.

		''')
	# Object Detection Function
def get_input_type(input_source, input_type="image"):
    # Load class names from the coco dataset
    class_file = './model/coco.txt'
    with open(class_file, 'rt') as file:
        class_names = file.read().rstrip('\n').split('\n')

    # Load model configurations and weights
    config_path = './model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = './model/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    if input_type == "image":
        # Process image
        img = cv2.imdecode(np.frombuffer(input_source.read(), np.uint8), cv2.IMREAD_COLOR)
        class_ids, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                if confidence > 0.5:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    if class_id <= len(class_names):
                        cv2.putText(img, class_names[class_id - 1].capitalize(),
                                    (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert image to RGB for Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Processed Image")

    elif input_type == "video":
        # Process live video stream
        cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
        stframe = st.empty()  # Placeholder for video stream
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.write("Failed to capture video")
                break

            class_ids, confs, bbox = net.detect(frame, confThreshold=0.5)
            if len(class_ids) != 0:
                for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                    if confidence > 0.5:
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                        if class_id <= len(class_names):
                            cv2.putText(frame, class_names[class_id - 1].capitalize(),
                                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()

# Streamlit App
st.title("Object Detection: Image and Video Live Stream")

# Sidebar for user selection
tab1, tab2 = st.tabs(["Image Capture", "Live Video Stream"])


with tab1:
	if tab1:
	    st.header("Capture an Image")
	    captured_image = st.camera_input("Take a picture")
	    if captured_image is not None:
	        get_input_type(captured_image, input_type="image")

with tab2:
	if tab2:
	    st.header("Live Video Stream")
	    st.write("Starting live video stream... (Press 'q' to exit)")
	    get_input_type(None, input_type="video")
	
