# Importing required libraries

import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("<center><font size=6><b>MICI, The Image Classifier</b></font></center>",unsafe_allow_html=True)

"""
### Introduction

Meet MICI (Machine Innovated to Classify Images), MICI is convolutional neural network model whose sole purpose is to 
be able to distinguish between two images. In simple terms, it is like a baby who tries to find and arrange a set of patterns
in order to distinguish between certain things. And just as a baby, its ability to classify images increases as it sees more and 
more images of the same kind. Right now, MICI is in need of images to increase its classification ability. Will you help MICI 
to get better? If yes, head on to the next section and follow the instructions.
"""

st.markdown("Hi, I am Chirag Gupta, the developer of MICI. I am a data science ethusiast who also interested in web development, game \
development and designing. You can connect with me \
on <a href='https://www.linkedin.com/in/chirag-gupta-359593218/'>LinkedIn</a>.",unsafe_allow_html=True)

"""
### Training MICI

1. All you have to do is create a folder in your google drive and name it anything you like. 

2. Inside that folder, create two sub folders that will contain images of two different entities.
(A minimum of 50 images per entity is recommended)

3. Make sure to name these folders according to the images that is stored in them.

4. Once you're done with this, enter the name of your main folder in the input box below and hit submit. 

5. MICI will then ask you permission to read the main folder. Grant the permission and wait for MICI to learn about the 
newly provided data.
"""

st.markdown("***")
folder_name = st.text_input("Name of the main folder:")
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Submit"):
    if not folder_name:
        st.error("Please enter the folder name first")

    else:
        # Configuring the API

        scope = ['https://www.googleapis.com/auth/drive.readonly']
        credentials_path = 'credentials.json'

        # Requesting access from user
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes=scope)
        creds = flow.run_local_server(port=0)

        # Building the service
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)

        # Retrieving the files
        results = service.files().list(q="mimeType='application/vnd.google-apps.folder' and name='{}'".format(folder_name), pageSize=10).execute()
        items = results.get('files', [])
        if items:
            print('Files:')
            for item in items:
                print(f'{item["name"]} ({item["id"]})')
        else:
            print('No files found.')