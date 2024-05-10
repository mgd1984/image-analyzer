import streamlit as st
import streamlit as st
import pandas as pd
import requests
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import av
import os
from pydub import AudioSegment
import math

from math import pi
from PIL import Image
from io import BytesIO
from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # # Configure the OpenAI API

# Enter you OpenAI API key here

selected_model = None
selected_key = None
with st.sidebar.expander("OpenAI Configuration"):
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'], index=1)
    selected_key = st.text_input("API Key", type="password")

client = OpenAI(api_key=selected_key) # Configure the OpenAI API

def convert_image_to_base64(image): # Convert san image to base64 format.
    # Convert the image to RGB mode
    image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str_jpeg = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str_png = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str_jpeg, img_str_png

def send_image_to_openai_vision_api(image, user_input, img_str): # Sends an image to the GPT-4 Turbo Vision API.
    
    # Define a list of voices
    voices = ['shimmer', 'alloy', 'echo', 'fable', 'onyx', 'nova']

    # Add a selectbox for the voices
    selected_voice = st.sidebar.selectbox("Voice", voices, index=0)
    
    # Intelligence level
    intelligence_level = st.sidebar.slider("Cognition", min_value=1, max_value=10, step=1, value=8, key="intelligence_level")

    # Model selection based on intelligence level
    if intelligence_level >= 8:
        selected_model = "gpt-4-turbo"
    elif intelligence_level >= 5:
        selected_model = "gpt-3.5-turbo"
    else:
        selected_model = "gpt-3.5-turbo"

    # Number of tokens based on intelligence level
    num_tokens = intelligence_level * 300

    # Temperature
    temperature = st.sidebar.slider("Creativity", min_value=0.1, max_value=1.0, step=0.1, value=0.5)
    
    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # user_input = st.text_input('Ask a question about the image:')


    submit_button = st.button('Submit Prompt', key="submit_prompt_button")
    


    # Send the image to the OpenAI Vision API
    with st.spinner("Analyzing"):
        if submit_button:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {selected_key}"
            }

            payload = {
                "model": selected_model,  # Use the selected model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": num_tokens  # Use the calculated number of tokens
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, stream=True)

            response_json = response.json()  # Extract the JSON response

            if response_json:
                # Extract the choices from the response
                choices = response_json.get('choices', [])

                # Display the choices
                for choice in choices:
                    st.markdown(f"**Image Analyzer:** {choice['message']['content']}", unsafe_allow_html=True)

                # Convert the response to speech
                response_text = " ".join([choice['message']['content'] for choice in choices])
                response_audio = client.audio.speech.create(
                    model="tts-1-hd",
                    voice=selected_voice,
                    input=response_text
                )

                # Save the audio to a file
                audio = AudioSegment.from_file(io.BytesIO(response_audio.read()), format="mp3")
                
                # audio = audio.speedup(playback_speed=1.2)
                audio_file = io.BytesIO()
                audio.export(audio_file, format='mp3')

                # Create a download link for the audio file
                audio_url = f'data:audio/mp3;base64,{base64.b64encode(audio_file.getvalue()).decode()}'

                # Display the audio player
                st.audio(audio_url)

            # # Print the response chunks
            # for chunk in response.iter_content(chunk_size=1024):
            #     print(chunk)
            #     print(chunk.choices[0].delta.content)
            #     print("****************")

            return response_json

# pages = ["Image Analyzer"]

# page = st.sidebar.selectbox("Pick a Page", pages)

# if page == "Image Analyzer": # Image Analyzer page allows users to analyze images using the GPT-4 Turbo Vision API.

with st.container():
            st.header("Image Analyzer")
            show_description = st.sidebar.checkbox("Show Description", value=True)
            if show_description:
                st.sidebar.write("""
                    The Advanced Image Analysis tool leverages the power of OpenAI's GPT-4 Turbo with advanced computer vision capabilities to analyze and process images with unparalleled accuracy and speed. Upload or capture an image, ask a question via the prompt window, then wait for GPT-4 Turbo to analyze the image and provide a detailed response. The tool also generates an audio response via the TTS-1-HD model with a choice of six different voices.
                    
                    We're way beyond [Hotdog or Not Hotdog](https://www.youtube.com/watch?v=vIci3C4JkL0) now.
                                
                    """)
            with st.expander("Upload an Image", expanded=True):
                uploaded_file = st.file_uploader("Upload an Image", type=['jpeg','png'])
            
            with st.expander("Take an Image", expanded=False):
                captured_image = st.camera_input(label="Capture an Image")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                user_input = st.text_input('Ask a question about the image:', key="image_question")
                img_str = convert_image_to_base64(image)
                response = send_image_to_openai_vision_api(image, user_input, img_str)
                # if response and 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                #     st.markdown(response['choices'][0]['message']['content'])
                # else:
                #     st.write("No content available")

            if captured_image is not None:
                # Read the image data from the captured image
                image = Image.open(captured_image)
                # Convert the image to an array format if necessary
                captured_image_array = np.array(image)
                # Now you can safely use captured_image_array or the image object for further processing
                user_input = st.text_input('Ask a question about the image:', key="image_question")
                img_str = convert_image_to_base64(image)
                response = send_image_to_openai_vision_api(image, user_input, img_str)
                