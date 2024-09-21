import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features
from real_time_recording import record_audio
import config

model = load_model(config.MODEL_PATH)

label_classes = np.load(config.LABEL_CLASSES_PATH)

print("Recording will start for 5 seconds...")

record_audio(config.RECORDED_AUDIO_PATH, duration=5)

print("Recording completed. Processing...")

features = extract_features(config.RECORDED_AUDIO_PATH)
features = features.reshape(1, -1, 1)

prediction = model.predict(features)
predicted_label_index = np.argmax(prediction)
predicted_emotion = label_classes[predicted_label_index]

print(f"Detected emotion: {predicted_emotion}")
