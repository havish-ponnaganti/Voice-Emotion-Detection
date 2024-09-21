import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from load_data import load_data
import config

def build_model(input_shape):
    """Build a CNN model."""
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, strides=4, activation='relu', input_shape=(input_shape, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=(4)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

features, labels, le = load_data(config.DATASET_PATH)

X = features[..., np.newaxis]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save(config.MODEL_PATH)
np.save(config.LABEL_CLASSES_PATH, le.classes_)
