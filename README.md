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

