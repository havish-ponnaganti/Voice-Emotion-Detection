import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from feature_extraction import extract_features

def load_data(dataset_path):
    """Load audio files and extract features."""
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    features, labels = [], []
    
    for actor in os.listdir(dataset_path):
        actor_folder = os.path.join(dataset_path, actor)
        for file_name in os.listdir(actor_folder):
            file_path = os.path.join(actor_folder, file_name)
            label = int(file_name.split("-")[2])
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(emotions.get(label, 'unknown'))
    
    features = np.array(features)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return features, labels, le
