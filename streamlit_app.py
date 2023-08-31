# Importing required libraries
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Variable initialization
label1_list = list()
label2_list = list()

# Creating a temporary directory to store uploaded files
upload_dir = "temp_upload"
os.makedirs(upload_dir, exist_ok=True)


# Functions

# Save uploaded files to the temporary directory
def save_uploaded_files(label_name, uploaded_files):
    label_dir = os.path.join(upload_dir, label_name.lower())
    os.makedirs(label_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(label_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())


# Title
st.markdown("<center><font size=6><b>MICI, The Image Classifier</b></font></center>",unsafe_allow_html=True)


# Introduction
"""
### Introduction

Meet MICI (Machine Innovated to Classify Images), MICI is convolutional neural network model whose sole purpose is to 
be able to distinguish between two images. In simple terms, it is like a baby who tries to find and arrange a set of patterns
in order to distinguish between certain things. And just as a baby, its ability to classify images increases as it sees more and 
more images of the same kind. Right now, MICI is in need of images to increase its classification ability. Will you help MICI 
to get better? If yes, head on to the next section and follow the instructions.
"""
st.markdown("Hi, I am Chirag Gupta, the developer of MICI. I am a data science ethusiast who is also interested in web development, game \
development and designing. You can connect with me \
on <a href='https://www.linkedin.com/in/chirag-gupta-359593218/'>LinkedIn</a>.",unsafe_allow_html=True)


# Uploading images
"""
### Training MICI
    
Note: A minimum of 50 images per label is recommended
"""

# User input of labels and images
st.markdown("***")
col1, col2 = st.columns(2)
label_name1 = col1.text_input("Label 1 name:")
label1_imgs = col1.file_uploader("Upload images for Label 1", type=[".jpeg",".png",".jpg"], accept_multiple_files=True)
label_name2 = col2.text_input("Label 2 name:")
label2_imgs = col2.file_uploader("Upload images for Label 2", type=[".jpeg",".png",".jpg"], accept_multiple_files=True)

# Submit button trigger
col1, col2, col3 = st.columns(3)
if col2.button("Submit", use_container_width=True):
    if (label_name1 and label1_imgs and label_name2 and label2_imgs):
        save_uploaded_files(label_name1, label1_imgs)
        save_uploaded_files(label_name2, label2_imgs)
    else:
        st.error("Please fill all input fields")


# Loading and preprocessing the uploaded images
uploaded_images = []
image_labels = []

for label_dir in os.listdir(upload_dir):
    if not os.path.isdir(os.path.join(upload_dir, label_dir)):
        continue

    label = label_name1 if label_dir == label_name1 else label_name2

    for root, dirs, files in os.walk(os.path.join(upload_dir, label_dir)):
        for file in files:
            file_path = os.path.join(root, file)
            image = load_img(file_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255.0  # Normalize pixel values to [0, 1]
            uploaded_images.append(image)
            image_labels.append(label)

# Converting the uploaded images and labels to NumPy arrays
uploaded_images = np.array(uploaded_images)
image_labels = np.array(image_labels)


# st.markdown("<center><font size=3><b>or</b></font></center><br>",unsafe_allow_html=True)

# with st.expander("Upload images from Google Drive"):
#     """
#     1. All you have to do is create a folder in your google drive and name it anything you like. 

#     2. Inside that folder, create two sub folders that will contain images of two different entities.
#     (A minimum of 50 images per entity is recommended)

#     3. Make sure to name these folders according to the images that is stored in them.

#     4. Once you're done with this, enter the name of your main folder in the input box below and hit submit. 

#     5. MICI will then ask you permission to read the main folder. Grant the permission and wait for MICI to learn about the 
#     newly provided data.
#     """
#     st.markdown("***")
#     folder_name = st.text_input("Name of the main folder:")
#     st.markdown("<br>", unsafe_allow_html=True)

#     if st.button("Submit"):
#         if not folder_name:
#             st.error("Please enter the folder name first")

#         else:
#             # Configuring the API

#             scope = ['https://www.googleapis.com/auth/drive.readonly']
#             credentials_path = 'credentials.json'

#             # Requesting access from user
#             flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes=scope)
#             creds = flow.run_local_server(port=0)

#             # Building the service
#             service = build('drive', 'v3', credentials=creds, cache_discovery=False)

#             # Retrieving the files
#             results = service.files().list(q="mimeType='application/vnd.google-apps.folder' and name='{}'".format(folder_name), pageSize=10).execute()
#             items = results.get('files', [])
#             if items:
#                 print('Files:')
#                 for item in items:
#                     print(f'{item["name"]} ({item["id"]})')
#             else:
#                 print('No files found.')