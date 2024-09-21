# Voice-Emotion-Detection
Project Overview
This project uses the RAVDESS dataset for speech emotion recognition. The code leverages Python-based machine learning libraries to extract features from audio data, train a model, and make predictions on emotional speech inputs. The following documentation will guide you through setting up your environment, running the code files, and understanding each file's functionality.

Tools & Technologies Used:
Python 3.8+
Librosa (Audio processing)
NumPy (Numerical computations)
Pandas (Data handling)
Scikit-learn (Machine learning)
TensorFlow (Deep learning)
PyAudio (Audio recording)
RAVDESS Dataset (Speech Emotion Recognition dataset)
1. Set up Virtual Environment
Open your terminal or command prompt.
Navigate to your project folder:
bash
Copy code
cd /path/to/your/project
Create a virtual environment:
bash
Copy code
python -m venv venv
Activate the virtual environment:
On Windows:
bash
Copy code
venv\Scripts\activate
On Mac/Linux:
bash
Copy code
source venv/bin/activate
2. Install Dependencies
Use the requirements.txt file to install the necessary libraries:

bash
Copy code
pip install -r requirements.txt
3. Project Files Explanation
1. config.py

This file defines configurations used throughout the project, such as paths to datasets, model parameters, and feature extraction settings. It centralizes configuration settings to avoid hardcoding values in different parts of the code.

2. feature_extraction.py

This script is responsible for extracting features from audio files, such as MFCCs (Mel Frequency Cepstral Coefficients) and chromagrams (representation of the 12 pitch classes). These features are critical for the model to recognize different emotions in speech.

Functionality:
Reads the audio files using librosa.
Extracts relevant audio features (MFCC, Chroma, Mel-Spectrogram, etc.).
Saves extracted features for training the model.
3. labelcheck.py

This file ensures that the labels (emotion categories) in the dataset are correctly assigned. It might check if the audio file names or paths are correctly linked with their respective emotions, ensuring clean and accurate labeling.

Functionality:
Verifies the mapping between file names and emotion labels.
Handles any errors in label assignment.
4. load_data.py

This script loads the dataset, processes it, and prepares it for training. It reads the audio files, applies feature extraction, and formats the data for model training.

Functionality:
Loads audio data from the RAVDESS dataset.
Applies feature extraction using functions from feature_extraction.py.
Splits the data into training and testing sets.
5. train_model.py

This script defines and trains the machine learning model. It utilizes TensorFlow/Keras to create a neural network that can recognize emotions from the extracted features.

Functionality:
Builds a deep learning model using TensorFlow.
Trains the model on the extracted features.
Saves the trained model for future predictions.
6. predict_emotion.py

This script loads the trained model and predicts emotions from new audio inputs. You can use this to test the model with unseen data.

Functionality:
Loads a pre-trained model.
Accepts audio files or real-time recordings for emotion prediction.
Outputs the predicted emotion.
7. real_time_recording.py

This script allows real-time recording of audio using PyAudio. After recording, it uses the trained model to predict the emotion present in the recorded speech.

Functionality:
Records audio using PyAudio.
Applies the same feature extraction methods used in train_model.py.
Predicts the emotion using the pre-trained model.
4. How to Run the Code
Step 1: Extract Features from Dataset
bash
Copy code
python feature_extraction.py
This script extracts the necessary audio features from the RAVDESS dataset and saves them for model training.
Step 2: Verify Labels (Optional) If you wish to check if the labels are correct:
bash
Copy code
python labelcheck.py
Step 3: Load and Prepare Data Load the dataset and prepare it for training:
bash
Copy code
python load_data.py
Step 4: Train the Model Train the deep learning model with the extracted features:
bash
Copy code
python train_model.py
Step 5: Make Predictions After training the model, you can use the predict_emotion.py file to predict emotions from new audio files:
bash
Copy code
python predict_emotion.py --file path_to_audio.wav
Step 6: Real-Time Emotion Prediction To use the real-time audio recording feature:
bash
Copy code
python real_time_recording.py
5. Requirements
Make sure the following packages are installed, as listed in the requirements.txt file:

plaintext
Copy code
librosa
numpy
pandas
scikit-learn
tensorflow
pyaudio
If any dependencies are missing or incorrect, you can install them manually using:

bash
Copy code
pip install <package_name>
6. General Workflow
Set up the environment (using virtual environment).
Install dependencies using the requirements.txt file.
Extract features from the dataset.
Train the model using the extracted features.
Use the trained model for emotion prediction on new audio or real-time recordings.
Example Commands:
To simplify the execution, you can create a run.sh script to automate the steps:

bash
Copy code
#!/bin/bash
source venv/bin/activate
python feature_extraction.py
python load_data.py
python train_model.py
This will automate the process of feature extraction, data loading, and model training in sequence.
