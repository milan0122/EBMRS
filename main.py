import streamlit as st 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import spotipy
import time
from spotipy.oauth2 import SpotifyClientCredentials
clientID ="cbc12782e54a46c59723938017a759bb"
secretID = "02ead26244ac44b7a335b2c284d19ae4"
st.markdown("""
    <style>
        .title-style{
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            padding: 20px;;   
        }
        .button-style> button {   
            padding: 15px;
            border-radius: 16px;
        }

        .button-style > button:hover {
           color: #005f73; 
        }
        .stButton>button{
            margin-left:300px;
        }
        .img-style{
            margin-left:200px;
        }
        .result{
            padding:20px;
            margin-bottom:20px;
            font-size:24px;
            display:flex;
            justify-content:center;
            algin-item:center;
            
        }
        

    </style>
""",unsafe_allow_html=True)     
#initialize 
Client_Credentials_Manager = SpotifyClientCredentials(client_id=clientID,client_secret=secretID)
sp = spotipy.Spotify(client_credentials_manager=Client_Credentials_Manager)
def get_track_details(track_id):
    track = sp.track(track_id)
    album_art_url = track["album"]["images"][0]["url"]
    song_name = track['name']
    artist_name = track['artists'][0]['name']
    return song_name,artist_name,album_art_url

song_df = pd.read_csv('notebook/music_moods.csv')

model = tf.keras.models.load_model('notebook/CNN_model_Checkpoint2.keras')
hascade_classifier = cv2.CascadeClassifier('notebook/frontal_face.xml')

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
        'neutral': ['Energetic'],
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
    st.write()
    st.markdown("<div class='result'>Recommended music</div>",unsafe_allow_html=True)
    st.write()
    for idx in range(0,len(top_songs),3):
        cols = st.columns(3)
        for i, track_id in enumerate(top_songs['id'][idx:idx+3]):
            if i <len(cols):
                popularity=top_songs['popularity'].iloc[idx+i]
                mood_type=top_songs['mood'].iloc[idx+i]
                song_name, artist_name, album_art_url = get_track_details(track_id)
                with cols[i]:
                    st.image(album_art_url,width=100)
                    st.markdown(f'  ⭐  {popularity/10}')
                    st.text(f"Title:  {song_name}")
                    st.text(f"Artist: {artist_name}")
                    st.text(f"type:   {mood_type}")
                    st.write()
                    st.write()
        

        


#use to recommend music on basis of emotion
def loadFile(uploaded_image):
    #convert PIL image to Open CV format
    image_path = np.array(uploaded_image)
    gray = cv2.cvtColor(image_path,cv2.COLOR_BGR2GRAY)
    #faces = hascade_classifier.detectMultiScale(image_path, 1.3, 5)
    faces = hascade_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0:
        st.warning("No face detected in the image")
        return
    for i,j,w,h in faces:
       # face_image = image_path[j:j+h, i:i+w]
        face_image = gray[j:j+h, i:i+w]
        cv2.rectangle(image_path,(i,j),(i+w,j+h),(255,0,0),2)
        face_image = cv2.resize(face_image,(48,48))
        feature = extract_features(face_image)
        #prediction
        pred = model.predict(feature)
        prediction_label =Emotion_Classes[pred.argmax()]
        st.markdown(f'<div class="result"> Predicted Emotion :{prediction_label}',unsafe_allow_html=True)
        img_rgb = cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
        st.markdown('<div class ="image-style">',unsafe_allow_html=True)
        st.image(img_rgb,width=500)
        st.markdown('</div>',unsafe_allow_html=True)
        

        #Recommend music
        Music_classifier(prediction_label)
  
def main(): 
    st.markdown('<div class="title-style">Emotion Based Music Recommendation System</div>',unsafe_allow_html=True)
    option = st.selectbox("Chose a photo or capture",['Select Photo','Select Camera'])
    if option == "Select Photo":
        with st.expander("Upload a photo"):
            upload_file = st.file_uploader('choose a photo',type=['jpeg','jpg','png'])
            # if upload_file is not None:
            #     img = Image.open(upload_file)
            #     st.image(img,caption="uploaded photo",use_column_width=200)

    else:
        upload_file = st.camera_input('Take a photo')
        # if upload_file is not None:
        #     img = Image.open(upload_file)
            #st.image(img,caption="photo captured",use_column_width=200)
    st.markdown('<div class="button-style">',unsafe_allow_html=True)
    if st.button("Recommend"):
        st.markdown('</div>',unsafe_allow_html=True)
        if upload_file is not None:
            placeholder=st.empty()
            progress = placeholder.progress(0,"Processing.....")
            time.sleep(1)
            progress.progress(50)
            time.sleep(1)
            progress.progress(100)
            image = Image.open(upload_file)
            loadFile(image)
            #st.success(result)
        else:
            st.warning("Please upload or take photo")

if __name__ == "__main__":
    main()
            
                
                
            


