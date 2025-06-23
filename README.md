#  Emotion-Based Music Recommendation System
## Project overview
This project demonstrates the integration of computer vision and deep learning (CNN) for emotion detection from facial expressions, and recommends music accordingly.

### Key Features
- Utilizes Convolutional Neural Networks (CNNs) for emotion classification.
- Trained on a dataset with ~27,000 images for training and ~7,000 for testing.
- Uses 3 convolutional layers and 2 dense layers.
- Softmax activation is used for multi-class classification.
- Adam optimizer is used to manage learning rate efficiently.
- Achieves approximately 60% accuracy on an imbalanced dataset.
- Integrated with Streamlit for a visual interface.
- Uses Haar Cascade classifier for frontal face detection.
- Recommends top 15 songs based on detected emotion.

## Program Structure
- app.py: This Streamlit application that lets you check your emotion via webcam (real-time) or through uploaded images.
- main.py: Emotion is detected in real-time, and you can capture a picture or upload one manually to get emotion-based music recommendations.

## SNAPSHOTS
Emotion Prediction and Evaluation
<table><tr><td><img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/sample.png" width="400" alt="Prediction through test image"></td> <td><img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/confusion_matrix.png" width="400" alt="Confusion matrix"></td> </tr> </table>
Model Training
<img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/accuracyvsloss.png" width="600" alt="Accuracy vs Loss">
Streamlit Web Interface
<table> <tr> <td><img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/Screenshot%202025-06-23%20at%2010.06.37.png" width="250" alt="UI 1"></td> <td><img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/Screenshot%202025-06-23%20at%2010.06.45.png" width="250" alt="UI 2"></td> <td><img src="https://github.com/milan0122/EBMRS/blob/949ab1dba1d6f23feec216c8843d051b40546f36/snapshots/Screenshot%202025-06-23%20at%2010.07.03.png" width="250" alt="UI 3"></td> </tr> </table>


