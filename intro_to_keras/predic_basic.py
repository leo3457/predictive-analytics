import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the model
model = keras.saving.load_model("final_model.keras")

# 2. Get 1 sample from x_test
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
img = x_test[0] # Grab the very first image
label = y_test[0]

# 3. Process and Predict
# Scale and add batch/channel dimensions
input_data = img.astype("float32") / 255
input_data = np.expand_dims(input_data, (0, -1)) # Shape becomes (1, 28, 28, 1)

predictions = model.predict(input_data)
confidence_scores = predictions[0] # This is a list of 10 probabilities

# 4. Print the "Confidence" breakdown
print(f"\nResults for the image (Actual Label: {label}):")
for digit, score in enumerate(confidence_scores):
    # Print how sure the AI is for each number 0-9
    print(f"Digit {digit}: {score*100:.2f}%")

# 5. Save the image so you can open it in VS Code
plt.imshow(img, cmap='gray')
plt.title(f"AI Guess: {np.argmax(confidence_scores)}")
plt.savefig("prediction_output.png") 
print("\nSuccess! Open 'prediction_output.png' in your folder to see the image.")