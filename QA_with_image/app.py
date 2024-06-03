import os
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai 
load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro-vision")
def get_gemini_response(input,image):
    if input != "":
        response = model.generate_content([input,image])
    else:
        response = model.generate_content(image)
    return response.text

# create app 

st.set_page_config(page_title="Image QA")
st.header("Image Q&A")
input = st.text_input("Input Prompt:",key="input")

uploaded_file = st.file_uploader("Choose an image..",type = ['jpeg','jpg','png'])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)

submit = st.button("Tell me about the image")

if submit:
    response = get_gemini_response(input,image)
    st.header("The response is")
    st.write(response)