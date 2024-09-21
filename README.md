<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>Speech Emotion Recognition using RAVDESS Dataset</h1>

<p>This project focuses on recognizing emotions from speech using the RAVDESS dataset. It utilizes Python-based machine learning libraries to extract audio features, train a model, and predict emotions from new audio inputs.</p>

<h2>Tools & Technologies Used</h2>

<ul>
<li>Python 3.8+</li>
<li>Librosa (Audio processing)</li>
<li>NumPy (Numerical computations)</li>
<li>Pandas (Data handling)</li>
<li>Scikit-learn (Machine learning)</li>
<li>TensorFlow (Deep learning)</li>
<li>PyAudio (Audio recording)</li>
<li>RAVDESS Dataset (Speech Emotion Recognition dataset)</li>
</ul>

<h2>0. Download Dataset</h2>
<p>RAVDESS Dataset : https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio</p>
Ensure that the dataset is correctly placed in the path defined in config.py.

<h2>1. Set up Virtual Environment</h2>

<ol>
<li>
<p>Navigate to your project folder:</p>

<pre><code>cd /path/to/your/project
</code></pre>
</li>
<li>
<p>Create a virtual environment:</p>

<pre><code>python -m venv venv
</code></pre>
</li>
<li>
<p>Activate the virtual environment:</p>

<ul>
<li>On Windows:

<pre><code>venv\Scripts\activate
</code></pre>
</li>
<li>On Mac/Linux:

<pre><code>source venv/bin/activate
</code></pre>
</li>
</ul>
</li>
</ol>

<h2>2. Install Dependencies</h2>

<pre><code>pip install -r requirements.txt
</code></pre>

<h2>3. Project Files Explanation</h2>

<ul>
<li><code>config.py</code>:  Defines project-wide configurations (dataset paths, model parameters, etc.).</li>
<li><code>feature_extraction.py</code>: Extracts audio features (MFCCs, chromagrams) from audio files.</li>
<li><code>labelcheck.py</code>:  Verifies the accuracy of emotion labels in the dataset.</li>
<li><code>load_data.py</code>: Loads, processes, and prepares the dataset for training.</li>
<li><code>train_model.py</code>:  Defines and trains the machine learning model.</li>
<li><code>predict_emotion.py</code>:  Loads the trained model and predicts emotions from new audio.</li>
<li><code>real_time_recording.py</code>:  Enables real-time audio recording and emotion prediction.</li>
</ul>

<h2>4. How to Run the Code</h2>

<h3>Step 1: Extract Features from Dataset</h3>

<pre><code>python feature_extraction.py
</code></pre>

<h3>Step 2: Verify Labels (Optional)</h3>

<pre><code>python labelcheck.py
</code></pre>

<h3>Step 3: Load and Prepare Data</h3>

<pre><code>python load_data.py
</code></pre>

<h3>Step 4: Train the Model</h3>

<pre><code>python train_model.py
</code></pre>

<h3>Step 5: Make Predictions</h3>

<pre><code>python predict_emotion.py --file path_to_audio.wav
</code></pre>

<h3>Step 6: Real-Time Emotion Prediction</h3>

<pre><code>python real_time_recording.py
</code></pre>

<h2>5. Requirements</h2>

<p>Make sure you have the following packages installed:</p>

<ul>
<li>librosa</li>
<li>numpy</li>
<li>pandas</li>
<li>scikit-learn</li>
<li>tensorflow</li>
<li>pyaudio</li>
</ul>

<p>You can install them manually using:</p>

<pre><code>pip install &lt;package_name&gt;
</code></pre>

<h2>6. General Workflow</h2>

<ol>
<li>Set up the virtual environment.</li>
<li>Install dependencies.</li>
<li>Extract features from the dataset.</li>
<li>Train the model.</li>
<li>Use the trained model for emotion prediction.</li>
</ol>

<h3>Example Commands:</h3>

<p>To automate the steps, create a <code>run.sh</code> script:</p>

<pre><code>#!/bin/bash
source venv/bin/activate
python feature_extraction.py
python load_data.py
python train_model.py
</code></pre>

<p>This script will automate feature extraction, data loading, and model training.</p>

<h3>Remember:</h3>

<p>Replace placeholders like <code>/path/to/your/project</code> and <code>path_to_audio.wav</code> with your actual file paths.</p>

</body>
</html>
