import cv2
import tensorflow as tf 
import numpy as np


#import models
model = tf.keras.models.load_model('models/fer_model.keras')
hascade_classifier = cv2.CascadeClassifier('models/frontal_face.xml')

def extract_features(image):
    feature = np.array(image)
    feature  = feature.reshape(1,48,48,1)
    return feature/255.0
Emotion_Classes = ['Angry', 
                  'Disgust', 
                  'Fear', 
                  'Happy', 
                  'Neutral', 
                  'Sad', 
                  'Surprise']

webcamp = cv2.VideoCapture(0)
while True:
    r,frame = webcamp.read()
    if not r:
        print("Error:Could not read frame")
        break
    frame = cv2.resize(frame,(600,500))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = hascade_classifier.detectMultiScale(frame,1.3,5)
    try:
        for (i,j,w,h) in faces:
            cv2.rectangle(frame,(i,j-50),(i+w,j+h+10),(0,255,0),2)
            image = gray[j:j+h, i:i+w]
            image = cv2.resize(image,(48,48))
            img = extract_features(image)
            print("Shape of img:",img.shape)
            
            pred = model.predict(img)
            print("Prediction Output:",pred)
            prediction_label = Emotion_Classes[pred.argmax()]
            cv2.putText(frame,'%s' %(prediction_label), (i+20, j-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2,cv2.LINE_AA)
            #cv2.putText(frame,'%s' %(prediction_label),(i+20,y-60),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255)),Lin
        cv2.imshow("Output",frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass
webcamp.release()
cv2.destroyAllWindows()