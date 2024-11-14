import streamlit as st 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
clientID ="cbc12782e54a46c59723938017a759bb"
secretID = "02ead26244ac44b7a335b2c284d19ae4"

#initialize 
Client_Credentials_Manager = SpotifyClientCredentials(client_id=clientID,client_secret=secretID)
sp = spotipy.Spotify(client_credentials_manager=Client_Credentials_Manager)
def get_track_details(track_id):
    track = sp.track(track_id)
    album_art_url = track["album"]["images"][0]["url"]
    song_name = track['name']
    artist_name = track['artists'][0]['name']
    return song_name,artist_name,album_art_url

song_df = pd.read_csv('data_moods.csv')

model = tf.keras.models.load_model('models/fer_model.keras')
hascade_classifier = cv2.CascadeClassifier('models/frontal_face.xml')

def extract_features(image):
    image_array = np.array(image)
    feature  = image_array.reshape(1,48,48,1)
    feature = feature/255.0
    return feature
Emotion_Classes = ['Angry', 
                  'Disgust', 
                  'Fear', 
                  'Happy', 
                  'Neutral', 
                  'Sad', 
                  'Surprise']


#function to classify the music according to mood 

def  Music_classifier(pred_class):
    # Mapping emotions to moods
    EMP = {
        'happy': ['Happy','Energetic'],
        'fear': ['Calm'],             
        'sad': ['Sad', 'Happy'],     
        'angry': ['Calm'],       
        'neutral': ['Happy','Energetic'],
        'surprise': ['Energetic','happy'],   
        'disgust': ['Sad']            
    }
    
    # Get the corresponding moods for the predicted emotion
    moods = EMP.get(pred_class.lower())
    
    if moods is None:
        print("Emotion not recognized.")
        return None

    play = song_df[song_df['mood'].isin(moods)]
    
    # Sort songs by popularity and select top 15
    top_songs = play.sort_values('popularity', ascending=False).head(15)
    for idx in range(0,len(top_songs),3):
        cols = st.columns(3)
        for i, track_id in enumerate(top_songs['id'][idx:idx+3]):
            if i <len(cols):
                song_name, artist_name, album_art_url = get_track_details(track_id)
                with cols[i]:
                    st.text(f"Title: {song_name}")
                    st.text(f"Artist: {artist_name}")
                    st.image(album_art_url,width=100)
                    st.write()
                    st.write()
        

        


#use to recommend music on basis of emotion
def loadFile(uploaded_image):
    #convert PIL image to Open CV format
    image_path = np.array(uploaded_image)
    gray = cv2.cvtColor(image_path,cv2.COLOR_RGB2GRAY)
    faces = hascade_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0:
        st.warning("No face detected in the image")
        return
    for i,j,w,h in faces:
        face_image = gray[j:j+h, i:i+w]
        cv2.rectangle(image_path,(i,j),(i+w,j+h),(255,0,0),2)
        face_image = cv2.resize(face_image,(48,48))
        feature = extract_features(face_image)
        #prediction
        pred = model.predict(feature)
        prediction_label =Emotion_Classes[pred.argmax()]
        st.title(f'prediction_label :{prediction_label}')
        img_rgb = cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
        st.image(img_rgb,use_container_width=200)
        

        #Recommend music
        Music_classifier(prediction_label)
        

st.text("Emotion-Based Music Recommendation System")
st.subheader("Choose an photo")


with st.expander("Upload a photo"):
    upload_file = st.file_uploader('choose a phtoto',type=['jpeg','jpg','png'])
    if upload_file is not None:
        img = Image.open(upload_file)
        st.image(img,caption="uploaded photo",use_container_width=200)
if st.button("Recommend"):
    if upload_file is not None:
        loadFile(img)
        


