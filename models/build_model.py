from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from efficientnet.tfkeras import EfficientNetB0

def build_efficientnetb0_model(input_size):
    # Load EfficientNetB0 with explicit input shape and without the top (classification layers)
    efficient_net = EfficientNetB0(weights='imagenet', input_shape=(input_size, input_size, 3), include_top=False)
    
    # Build the model
    model = Sequential([
        efficient_net,
        GlobalAveragePooling2D(),  # Use global pooling to flatten the output
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])
    
    return model
