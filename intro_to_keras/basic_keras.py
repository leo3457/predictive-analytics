import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

# Notes that Keras should only be imported after the backend has been configured. The backend cannot be changed once the package is imported
# If you want to change the backend, you must restart the Python interpreter and set the environment variable before importing Keras.

import keras

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0,1] range
# Reshape images to [0, 1] range like this
x_train = x_train.astype("float32") / 255
# Use float32 and divide by 255 because pixels are ints b/n 0(black) and 255(white). Dividing by 255 maps them to range b/n 0.0 and 1.0
x_test = x_test.astype("float32") / 255 
# If you want to center around 0 instead (eg temperature), use z-score: x_train_centered = (x_train - mean) / std

# Make sure images have shape (28, 28, 1)
# These are specific to black and white images. For colour, there is no need.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# For a color image that is 32x32 pixels
# input_shape = (32, 32, 3) 

# model = keras.Sequential([
#     keras.layers.Input(shape=input_shape),
#     keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#     # ... rest of the model
# ])

# Always good practice to print shape
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Model parameters
# Creating 10 "output doors" at the end of the model
# Future Applications: Medical: 2 categories (Cancerous vs Healthy), Self-Driving (Stop sign, Pedestrian, Green light, Red light)
num_classes = 10


# This is just a Python label
input_shape = (28, 28, 1)

# Three type of model-building options that Keras offers:
# The Sequential API (what we use below)
# The Functional API (most typical)
# Writing your own models yourself via subclassing (for advanced use cases)

model = keras.Sequential(
    [
        # This explicitly defines the starting point and shape of the data entering the model
        keras.layers.Input(shape=input_shape),

        # 64: The layer creates 64 different specialized filters. 
        # kernel_size=(3, 3): Each filter is a $3 \times 3$ pixel window that slides across the image.
        # relu: The "Rectified Linear Unit" activation function which helps the model learn complex patterns by ignoring negative values
        # The First Layer: Looks for very simple things like straight lines or dots.
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        # The Second Layer: Looks at the lines from the first layer and starts seeing shapes, like circles or corners
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        # Shrinks the image to highlight the most important features and save memory
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),

        # Instead of keeping a grid of numbers, this averages the entire feature map into a single number per filter. 
        # This prepares the data for the final classification. Flattens that 3D data into a 1D list
        keras.layers.GlobalAveragePooling2D(),

        # Randomly "turns off" 50% of the neurons during training. Prevents overfitting
        keras.layers.Dropout(0.5),

        # num_classes: The final layer has 10 neurons, one for each digit. softmax: This converts the output into probabilities
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# The compile() method is essentially where you "configure" the brain of your model before it starts learning. 
# While the layers define the structure, compile() defines the rules of the game
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

# --- Study Session ---
# This section of the code handles the actual "study session" for your AI and establishes a safety net 
# to ensure you don't lose progress or over-train the model

# 1. Training Parameters
# This tells the model to look at 128 images at a time before updating its internal math.
batch_size = 128
# This is the maximum number of times the model will repeat the entire "textbook" (dataset).
epochs = 20

# 2. The "Safety Net" (Callbacks)
# Callbacks are functions that run automatically at the end of every epoch to perform specific tasks.
callbacks = [
    # This saves a file (e.g., model_at_epoch_5.keras) after every round. 
    # If your computer crashes at epoch 10, you still have the version from epoch 9
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    # This monitors the "Practice Quiz" (val_loss). If the model stops getting smarter for 2 rounds (patience=2), 
    # Keras kills the training early to save time and prevent the model from just memorizing the data
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

# 3. Executing the Training
# model.fit(...): This starts the training process using the training data (x_train, y_train).
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    # Keras automatically hides 15% of your training data to use for those "Practice Quizzes"
    validation_split=0.15,
    callbacks=callbacks,
)

# Once training is totally finished, this runs the "Final Exam" on the test data (x_test, y_test) to give you a final accuracy score
score = model.evaluate(x_test, y_test, verbose=0)

