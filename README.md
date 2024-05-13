# Image Analyzer and Generator

The best way to learn AI is by building. In that spirit, this project is a Streamlit web app that uses OpenAI's GPT-4 Turbo and DALL-E models to analyze and generate images. The Image Analyzer tool allows users to upload or capture an image, ask a question, and receive a detailed response from the GPT-4 Turbo model. The Image Generator tool uses the DALL-E model to generate images based on user prompts. The application also includes text-to-speech (TTS) functionality to generate audio responses using the TTS-1-HD model.

## Features

1. **Image Analyzer**: Upload or capture an image, ask a question via the prompt window, then wait for GPT-4 Turbo to analyze the image and provide a detailed response. The tool also generates an audio response via the TTS-1-HD model with a choice of six different voices.

2. **Image Generator & Comparison Tools**: Enter a prompt, select the image size and quality, and specify the number of images to generate. The DALL-E model will generate images based on the prompt. The tool also includes a comparison feature to compare the generated images side by side.

## Setup

1. Clone the repository.
2. Install the required Python packages using pip:

```sh
pip install -r requirements.txt
```

3. Run the Streamlit app:

```sh
streamlit run app.py
```

## Usage

1. Select a page from the sidebar: "Image Analyzer" or "Image Generator".
2. For the Image Analyzer, upload or capture an image and ask a question about the image.
3. For the Image Generator, enter a prompt, select the image size and quality, and specify the number of images to generate.
4. Click the 'Submit' button to generate the images or analyze the uploaded/captured image.

## OpenAI Configuration

You need to provide your OpenAI API key to use the application. Enter your API key in the "OpenAI Configuration" section in the sidebar.

## Dependencies

This project uses several Python libraries including Streamlit, OpenAI, pandas, requests, base64, io, matplotlib, numpy, av, os, pydub, math, cv2, and PIL.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the terms of the MIT license.