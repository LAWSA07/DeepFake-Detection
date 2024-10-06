import os
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
checkpoint_filepath = './tmp_checkpoint'
test_path = './split_dataset/test'

# Load the best model
model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Set up test data generator
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(128, 128), batch_size=1, class_mode=None, shuffle=False
)

# Generate predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)

# Save predictions to CSV
test_results = pd.DataFrame({"Filename": test_generator.filenames, "Prediction": predictions.flatten()})
test_results.to_csv("predictions.csv", index=False)
print(test_results)
