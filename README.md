# Speech Emotion Recognition using RAVDESS Dataset

This project leverages the RAVDESS dataset for Speech Emotion Recognition. The code utilizes Python-based machine learning libraries to extract features from audio data, train a model, and make predictions on emotional speech inputs. This documentation will guide you through setting up the environment, running the code, and understanding the functionality of each file.

## Tools & Technologies Used

- **Python 3.8+**
- **Librosa** (Audio processing)
- **NumPy** (Numerical computations)
- **Pandas** (Data handling)
- **Scikit-learn** (Machine learning)
- **TensorFlow** (Deep learning)
- **PyAudio** (Audio recording)
- **RAVDESS Dataset** (Speech Emotion Recognition dataset)

## 1. Set up Virtual Environment

1. Open your terminal or command prompt.
2. Navigate to your project folder:

    ```bash
    cd /path/to/your/project
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On Mac/Linux:

      ```bash
      source venv/bin/activate
      ```

## 2. Install Dependencies

Use the `requirements.txt` file to install the necessary libraries:

```bash
pip install -r requirements.txt
```
# 3. Project Files Explanation

## Project Structure

### 1. `config.py`
This file defines configuration settings used throughout the project to avoid hardcoding values. It includes paths to datasets, model parameters, and feature extraction settings.

**Key functionality:**
- Centralizes configuration settings such as dataset paths and model parameters.
- Ensures consistency in settings across all scripts.

### 2. `feature_extraction.py`
This script is responsible for extracting essential features from audio files, such as **MFCCs (Mel Frequency Cepstral Coefficients)** and **chromagrams**. These features are crucial for training the model to recognize various emotions in speech.

**Key functionality:**
- Reads audio files using the `librosa` library.
- Extracts relevant audio features such as MFCC, Chroma, and Mel-Spectrogram.
- Saves the extracted features for model training.

### 3. `labelcheck.py`
This script ensures that the labels (emotion categories) in the dataset are correctly assigned. It verifies the correctness of labels to ensure clean and accurate data for training.

**Key functionality:**
- Verifies the mapping between file names and emotion labels.
- Handles potential errors in label assignment.

### 4. `load_data.py`
This script is responsible for loading the RAVDESS dataset and preparing it for training. It loads the audio data, applies feature extraction, and formats it for model training.

**Key functionality:**
- Loads audio data from the RAVDESS dataset.
- Applies feature extraction using the `feature_extraction.py` script.
- Splits the data into training and testing sets.

### 5. `train_model.py`
This script defines the machine learning model and handles the training process. The model is built using **TensorFlow/Keras** to recognize emotions from the extracted audio features.

**Key functionality:**
- Builds a deep learning model using TensorFlow.
- Trains the model on extracted features.
- Saves the trained model for future predictions.

### 6. `predict_emotion.py`
This script loads the pre-trained model and predicts emotions from new audio inputs. It can be used to test the model with unseen data.

**Key functionality:**
- Loads a pre-trained model.
- Accepts audio files or real-time recordings for emotion prediction.
- Outputs the predicted emotion.

### 7. `real_time_recording.py`
This script enables real-time audio recording using **PyAudio**. After recording, the script uses the trained model to predict the emotion in the recorded speech.

**Key functionality:**
- Records audio using PyAudio.
- Applies the same feature extraction methods used during training.
- Predicts the emotion from the recorded speech using the pre-trained model.

# 4. How to Run the Project

1. **Install dependencies**: Ensure you have all the required libraries installed by running:
   ```bash
   pip install -r requirements.txt


