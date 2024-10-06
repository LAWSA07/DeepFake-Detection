import os
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.build_model import build_efficientnetb0_model
from utils.data_loader import load_dataset
import pandas as pd

dataset_path = './data/'
checkpoint_filepath = './tmp_checkpoint'
tmp_debug_path = './tmp_debug'

# Create necessary directories if not exist
os.makedirs(tmp_debug_path, exist_ok=True)
os.makedirs(checkpoint_filepath, exist_ok=True)

input_size = 128
batch_size_num = 32
num_epochs = 20

# Load video data (real and fake)
real_data, real_labels = load_dataset(os.path.join(dataset_path, 'real_videos'), label=1, frame_skip=10)
fake_data, fake_labels = load_dataset(os.path.join(dataset_path, 'fake_videos'), label=0, frame_skip=10)

# Combine and shuffle data
X = np.concatenate((real_data, fake_data), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

# Shuffle data
shuffle_idx = np.random.permutation(len(y))
X, y = X[shuffle_idx], y[shuffle_idx]

# Split data into train, validation, and test sets
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Normalize pixel values
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

# Build EfficientNetB0 model
model = build_efficientnetb0_model(input_size)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks
custom_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
    ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'best_model.keras'), 
                    save_best_only=True, monitor='val_loss', mode='min', verbose=1)
]

# Train the model
history = model.fit(
    X_train, y_train, epochs=num_epochs, batch_size=batch_size_num, validation_data=(X_val, y_val),
    callbacks=custom_callbacks
)

# Load the best model and evaluate on the test set
best_model = models.load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Generate predictions on test data
preds = best_model.predict(X_test)
test_results = pd.DataFrame({"Prediction": preds.flatten(), "Actual": y_test})
print(test_results)
