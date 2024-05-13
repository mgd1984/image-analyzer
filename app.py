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
import cv2
from streamlit_image_comparison import image_comparison

from math import pi
from PIL import Image
from io import BytesIO
from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # # Configure the OpenAI API

# Enter you OpenAI API key here

selected_model = None
selected_key = None
with st.sidebar.expander("OpenAI Configuration"):
    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo','gpt-4o'], index=3)
    selected_key = st.text_input("API Key", type="password")
    st.write("[Get an OpenAI API Key](https://platform.openai.com/api-keys)")

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

pages = ["Image Analyzer", "Image Generator"]

page = st.sidebar.selectbox("Pick a Page", pages)

if page == "Image Analyzer": # Image Analyzer page allows users to analyze images using the GPT-4 Turbo Vision API.

    with st.container():
                st.header("Image Analyzer")
                show_description = st.sidebar.checkbox("Show Description", value=True)
                if show_description:
                    st.sidebar.write("""
                        The Advanced Image Analysis tool leverages the power of OpenAI's GPT-4 Turbo (and now GPT-4o) with advanced computer vision capabilities to analyze and process images with unparalleled accuracy and speed. Upload or capture an image, ask a question via the prompt window, then wait for the model to analyze the image and provide a detailed response. The tool also generates an audio response via the TTS-1-HD model with a choice of six different voices.
                        
                        We're way beyond [Hotdog or Not Hotdog](https://www.youtube.com/watch?v=vIci3C4JkL0) now.
                                    
                        """)
                with st.expander("Upload an Image", expanded=True):
                    uploaded_file = st.file_uploader("Upload an Image", type=['jpeg','png'])
                
                # with st.expander("Take an Image", expanded=False):
                #     captured_image = st.camera_input(label="Capture an Image")

                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    user_input = st.text_input('Ask a question about the image:', key="image_question")
                    img_str = convert_image_to_base64(image)
                    response = send_image_to_openai_vision_api(image, user_input, img_str)
                    # if response and 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    #     st.markdown(response['choices'][0]['message']['content'])
                    # else:
                    #     st.write("No content available")

                # if captured_image is not None:
                #     # Read the image data from the captured image
                #     image = Image.open(captured_image)
                #     # Convert the image to an array format if necessary
                #     captured_image_array = np.array(image)
                #     # Now you can safely use captured_image_array or the image object for further processing
                #     user_input = st.text_input('Ask a question about the image:', key="image_question")
                #     img_str = convert_image_to_base64(image)
                #     response = send_image_to_openai_vision_api(image, user_input, img_str)
elif page == "Image Generator": # Image Generator page allows users to generate images using OpenAI's DALL-E model.

        # Check permissions
            st.title("Image Generator")
            show_description = st.sidebar.checkbox("Show Description", value=True)
            if show_description:
                st.sidebar.write("""
                    The Image Generation tool allows you to generate images using OpenAI's DALL-E model. Simply enter a prompt, select the image size and quality, and the tool will automatically generate images based on your input. Customize the number of images to generate and explore the creative possibilities with the DALL-E model. Experience the power of AI-driven image generation with the Image Generation tool.
                    """)
                prompts = [
                    "A breathtaking sunset over a pristine beach",
                    "A mysterious figure walking through a foggy forest",
                    "A bustling cityscape with towering skyscrapers",
                    "A majestic waterfall cascading down a rocky cliff",
                    "A vibrant street market filled with exotic fruits and spices",
                    "A group of adventurers exploring a hidden treasure cave",
                    "A serene mountain lake reflecting the starry night sky",
                    "A futuristic cityscape with flying cars and holographic displays",
                    "A magical garden with talking animals and enchanted flowers",
                    "A thrilling roller coaster ride with twists and turns"
                ]

                # prompt = random.choice(prompts)
                prompt = st.text_input("Enter a Prompt")
            with st.expander("Image Generation Options", expanded=False):
                size = st.selectbox("Image size", ["1024x1024", "1024x1792", "1792x1024"])
                quality = st.selectbox("Image quality", ["standard", "hd"])
                # n = st.slider("Number of images to generate", 1, 10, 1)
                if st.button('Submit', key="generate_image_button"):
                    with st.spinner("Generating images..."):
                        response = client.images.generate(
                            model="dall-e-3",
                            prompt=prompt, 
                            size=size,
                            quality=quality,
                            n=n,
                        )

                        image_urls = [image.url for image in response.data]
                        image_url = image_urls[0]

                        # Display the generated images
                        for i, image_url in enumerate(image_urls):
                            image_path = f"generated_image_{i+1}.png"
                            image_data = requests.get(image_url).content

                            # Check if the image was saved correctly
                            if not image_data:
                                st.write(f"Failed to save image {image_url}")
                                continue

                            with open(image_path, "wb") as f:
                                f.write(image_data)

                            try:
                                image = Image.open(image_path)
                            except Exception as e:
                                st.write(f"Failed to load image {image_path}: {e}")
                                continue

                            st.image(image, caption='Generated Image', use_column_width=True)
                            image_bytes = cv2.imencode('.png', np.array(image))[1].tobytes()
                            st.download_button("Download Image", image_bytes, "generated_image.png", mime="image/png")


                if st.button('Generate Image Variation w/ Dalle', key="generate_variation_button"):
                    image_path = f"generated_image_1.png"  # Define the image_path variable
                    response = client.images.create_variation(
                        image=open(image_path, "rb"),
                        n=1,
                        size=size
                    )
                    variation_urls = [variation.url for variation in response.data]
                    for i, variation_url in enumerate(variation_urls):
                        variation_path = f"generated_variation_{i+1}.png"
                        variation_data = requests.get(variation_url).content
                        if not variation_data:
                            st.write(f"Failed to save variation {variation_url}")
                            continue
                        with open(variation_path, "wb") as f:
                            f.write(variation_data)
                        try:
                            variation_image = Image.open(variation_path)
                        except Exception as e:
                            st.write(f"Failed to load variation {variation_path}: {e}")
                            continue
                        st.image(variation_image, caption=f"Generated Variation {i+1}", use_column_width=True)
                        variation_bytes = cv2.imencode('.png', np.array(variation_image))[1].tobytes()
                        st.download_button(f"Download Variation {i+1}", variation_bytes, f"generated_variation_{i+1}.png", mime="image/png")
                        
                        # Add custom CSS to constrain the size of the container
                        css_style = """
                        <style>
                        .container {
                            display: flex;
                            justify-content: center;
                            max-width: 90%;
                            overflow-x: auto;  /* Enables horizontal scrolling if content is wider than screen */
                            box-sizing: border-box; /* Includes padding and border in the element's total width and height */
                        }
                        .image-comparison {
                            max-width: 80%;  /* Ensures the image comparison does not exceed the container width */
                            height: auto;  /* Maintains the aspect ratio of the image */
                        }
                        </style>
                        """

                        st.markdown(css_style, unsafe_allow_html=True)  # Apply the custom CSS

                        # Wrap the image comparison in a div with the 'container' class
                        st.markdown('<div class="container">', unsafe_allow_html=True)
                        image_comparison(
                            img1=variation_image,
                            label2="Dalle 3",
                            img2=Image.open(image_path),
                            label1="Dalle 2",
                            # width=800,  # You might need to adjust or remove this width setting depending on your needs
                            starting_position=50,
                            show_labels=True,
                            make_responsive=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)  # Close the div


                        # # Compare original image and variation
                        # st.markdown('<div style="display: flex; justify-content: center; overflow-x: auto;">', unsafe_allow_html=True)
                        # image_comparison(
                        #     img1=variation_image,
                        #     label2=f"Dalle 3",
                        #     img2=Image.open(image_path),
                        #     label1="Dalle 2",
                        #     width=800,
                        #     starting_position=50,
                        #     show_labels=True,
                        #     make_responsive=True,
                        # )
                        # st.markdown('</div>', unsafe_allow_html=True)

                # else:
                #     st.error("Failed to generate image variation. Please try again.")

            # response = generate_image_with_dalle(prompt, size, quality, n)
            # if response.status_code != 200:
            #     st.error("Failed to generate image. Please try again.")
else:
    st.error("You do not have permission to access the Image Generator.")
