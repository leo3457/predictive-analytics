import keras
import numpy as np
import tensorflow as tf

# 1. Load the "Frozen Brain"
model = keras.models.load_model('my_character_model.keras')

# 2. You still need your 'translator' (same as before)
text = "This is GeeksforGeeks a software training institute"
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# 3. Predict immediately!
start_seq = "This is G"
generated_text = start_seq
seq_length = 3

for i in range(20):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    prediction = model.predict(x_one_hot, verbose=0)
    next_char = index_to_char[np.argmax(prediction)]
    generated_text += next_char

print(f"Prediction:")
print(generated_text)