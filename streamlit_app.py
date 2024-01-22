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
import shutil
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Rescaling
from keras.models import load_model


# Variable initialization
label1_list = list()
label2_list = list()
upload_dir = "temp_upload"
if os.path.exists(upload_dir):
    shutil.rmtree(upload_dir)
os.makedirs(upload_dir)
DATASET_PATH = "temp_upload"
IMG_RES = (224, 224)
BATCH_SIZE = 32
EXECUTE_CNN = False


# Functions

# Save uploaded files to the temporary directory
def save_uploaded_files(label_name, uploaded_files):
    label_dir = os.path.join(upload_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join(label_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
            


# Title
st.markdown("<center><font size=6><b>MICI, The Image Classifier</b></font></center>",unsafe_allow_html=True)


# Introduction
"""
### Introduction

Meet MICI (Machine Innovated to Classify Images), MICI is a convolutional neural network model whose sole purpose is to 
be able to distinguish between two images. In simple terms, it is like a baby who tries to identify and arrange a set of patterns
in order to distinguish between certain things. And just as a baby, its ability to classify images increases as it sees more and 
more images of the same kind. Right now, MICI is in need of images to increase its classification ability. Will you help MICI 
to get better? If yes, head on to the next section and fill in the fields.
"""
st.markdown("Hi, I am Chirag Gupta, the developer of MICI. I am a data science ethusiast who is also interested in software development \
and designing. You can connect with me on \
<a href='https://www.linkedin.com/in/chirag-gupta-359593218/'>LinkedIn</a>.",unsafe_allow_html=True)


# Uploading images
"""
### Training MICI
    
Note: A minimum of 50 images per label is recommended
"""

# User input of labels and images
with st.form("inputform", clear_on_submit=True):
    col1, col2 = st.columns(2)
    label_name1 = col1.text_input("Label 1 name:", key='labelname1').lower()
    label1_imgs = col1.file_uploader("Upload images for Label 1", type=[".jpeg",".png",".jpg"], accept_multiple_files=True, key='labelimgs1')
    label_name2 = col2.text_input("Label 2 name:", key='labelname2').lower()
    label2_imgs = col2.file_uploader("Upload images for Label 2", type=[".jpeg",".png",".jpg"], accept_multiple_files=True, key='labelimgs2')

    # Submit and Clear button trigger
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        submit = st.form_submit_button("Submit", use_container_width=True)
        if submit:
            if (label_name1 and label1_imgs and label_name2 and label2_imgs):
                save_uploaded_files(label_name1, label1_imgs)
                save_uploaded_files(label_name2, label2_imgs)
                EXECUTE_CNN = True
            else:
                st.error("Please fill all input fields")
    with col3:
        clear = st.form_submit_button("Clear", use_container_width=True)
        if clear:
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                os.makedirs(upload_dir)
    

# Building the CNN model
if EXECUTE_CNN:
    num_images = len(label1_imgs) + len(label2_imgs)
    if num_images >= 32:
        BATCH_SIZE = 32
    elif num_images >=16:
        BATCH_SIZE = 16
    elif num_images >=8:
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 4

    train_batches = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        subset="training",        # Specifying that this is the training set
        validation_split=0.2,     # Leave 20% for the validation set
        shuffle=True,             # For random shuffling of data
        seed=123,                 # Needs to be specified for shuffling
        label_mode="binary",      # To encode labels as binary vectors
        batch_size=BATCH_SIZE,
        image_size=IMG_RES
    )

    validation_batches = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        subset="validation",
        validation_split=0.2,
        shuffle=True,
        seed=123,
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=IMG_RES
    )

    # test_batches = tf.keras.utils.image_dataset_from_directory(
    #     DATASET_PATH,
    #     subset="validation",
    #     validation_split=0.2,
    #     shuffle=True,
    #     seed=123,
    #     label_mode="binary",
    #     batch_size=BATCH_SIZE,
    #     image_size=IMG_RES
    # )

    # print(train_batches.classes)
    # print(list(train_batches.class_indices.keys())[0])

    data_augmentation = Sequential(
    [
        keras.layers.RandomFlip("horizontal",input_shape= IMG_RES + (3,)),
        keras.layers.RandomContrast(0.5),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
    ])

    model = Sequential([
        data_augmentation,
        Rescaling(1./255, input_shape=IMG_RES + (3,)),
        Conv2D(32, 3, activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, 3, activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, 3, activation='relu'),
        MaxPool2D((2, 2)),
        # Conv2D(128, 3, activation='relu'),
        # MaxPool2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    print(model.summary())
    
    # saving the model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="mici.h5",
                                                    mode='min', 
                                                    monitor='val_loss',
                                                    save_best_only=True)
    # early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        mode='min',
        restore_best_weights=True)

    # Model training
    history = model.fit(train_batches, 
                        validation_data=validation_batches, 
                        epochs=30,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[checkpoint, early_stop]).history

    # Loading the model
    model = load_model('mici.h5')
    class_names = train_batches.class_names
    print(class_names)
    # Retrieve a batch of images from the test set
    image_batch, label_batch = validation_batches.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    print(label_batch)
    print(predictions)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(image_batch[i].astype("uint8"))
        ax.set_title(class_names[predictions[i]])
        ax.axis("off")
    st.pyplot(fig)


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