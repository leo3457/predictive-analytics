🧠 Keras 3: Image Classification Project (MNIST)
📋 Executive Summary
This project involved building a deep learning model to identify handwritten digits (0–9). We used Keras 3, a modern framework that allows the same code to run on TensorFlow, JAX, or PyTorch. We successfully navigated the full pipeline: data preparation, model architecture, training with safety "callbacks," and saving the model for future use.

🛠️ Key Concepts & Rules
1. The Backend-Agnostic Rule
What is a Backend? It is the "engine" (TensorFlow, JAX, or PyTorch) that does the math.
The Import Rule: You must tell Keras which engine to use before you import it. Once imported, it is locked.
Python

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras


2. Data "Splitting" & Preparation
Training Set: The "textbook" the AI studies from.
Validation Split: A "practice quiz" (usually 15%) taken during training to check if the AI is actually learning or just memorizing.
Test Set: The "final exam" used only after training is finished to get a true accuracy score.
Scaling: We divide pixel values by 255 to turn them into small decimals (0.0 to 1.0). This makes the math stable.
Reshaping: We use expand_dims to add a "1" to the end of the shape (e.g., 28, 28, 1). This tells the model it is a grayscale image.

🏗️ The Model "Assembly Line" (Sequential API)
We stacked layers in a specific order to process images:
Input Layer: The entry point for $28 \times 28 \times 1$ data.
Conv2D: The "Eyes" that scan for features like edges and curves.
MaxPooling2D: Shrinks the image to highlight the most important parts.
GlobalAveragePooling: Flattens the 3D image data into a 1D list for the "brain".
Dropout: Prevents "overfitting" by randomly turning off half the neurons during training.
Dense (Output): 10 doors (digits 0–9) using Softmax to give us probabilities.

📉 Training & Automation
Compile: We configured the model with the Adam optimizer and Crossentropy loss.
Epochs: The number of times the AI reads the entire dataset (we used 20).
ModelCheckpoint: Automatically saves the model after every epoch so we don't lose progress.
EarlyStopping: Automatically stops training if the model stops getting smarter, preventing wasted time.

💾 Saving and Using the Model (Inference)
Once the model is trained, we don't need to train it again. We save the "brain" to a file.
Save: model.save("final_model.keras").
Load: model = keras.saving.load_model("final_model.keras").
Predict: predictions = model.predict(new_images).
The output is 10 probabilities. Use np.argmax() to find the highest one (the AI's guess).

❓ Frequently Asked Questions
Q: Why scale pixels to [0, 1]?
A: Neural networks learn better with small numbers. If numbers are too big (like 255), the math can become unstable.
Q: Why use "Sparse" Categorical Crossentropy?
A: We use "Sparse" because our labels are simple integers (0, 1, 2...). If our labels were lists of zeros (One-Hot Encoded), we would just use regular "Categorical Crossentropy".
Q: What does MaxPooling actually do?
A: It looks at a small window of pixels and keeps only the brightest one. This makes the model faster and helps it recognize shapes regardless of where they are in the image.
Q: Why can't I see the images with plt.show()?
A: Some terminals don't support "pop-up" windows. In those cases, use plt.savefig("result.png") to save the image as a file you can click on in VS Code.
