##
# Source origin
# - ChatGPT: Base python code, fixes, improvements
# - Christian: Feature requirements, data structures
##

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ------------------------------------------------------------------------------

# DATA

# Raw values
# - Index = 0 is also expected output
data = np.array([
    [1, "ASTEROID", 100, 100],
    [0, "ASTEROID", 0, 100],
    [0, "ASTEROID", 100, 0],
    [0, "PARTICLE", 100, 100],
    [0, "PARTICLE", 0, 100],
    [0, "PARTICLE", 100, 0],
    [1, "ASTEROID", 0, 0],
])


# ------------------------------------------------------------------------------

# TRANSFORM

# - Raw data index 1-3 (ignore index 0)
# - Raw data index 0
inputs = data[:, 1:]
outputs = data[:, 0].astype(int)
#print(inputs)
#print(outputs)


# Strings become integers (here: Asteroid=0, Particle=1)
# - One-hot encode the 'category' column (column index 0)
# - Transformation creates columns for ALL unique input values (ASTEROID, PARTICLE)
# -- This is why there are now 4 inputs instead 3 (first row = data as int, 2nd = inverse)
# -- It is like a matrix where each feature is combined with each other
# -- It is used to prevent models from thinking of ordinal relationships (A>B)
# --- Given all values as matrix cancel out each other, model can focus on other data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="passthrough"
)
inputs = np.array(ct.fit_transform(inputs), dtype=float)
#print(inputs)


# ------------------------------------------------------------------------------

# MODEL

# Hidden layers = Between in/out, not visible/usable?
# - 8 neurons with 4 inputs
# -- category, categoryInverse, pos_x_item, pos_x_player
# - ReLU for complex pattern learning (hidden layers?)
# - Sigmoid maps output to 0-1 value ranges
# Todo: Test what happens if middle layer is removed (might not be needed, optimizes it)
model = Sequential()
model.add(Dense(8, input_dim=4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Choices are popular for this problem solving according to ChatGPT 4
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(inputs, outputs, epochs=1000, batch_size=2)


# ------------------------------------------------------------------------------
# PREDICTIONS

def predict_collision(category: str, pos_x_item: int, pos_x_player: int):
    # Convert the input data into a NumPy array
    input_data = np.array([[category, pos_x_item, pos_x_player]])
    #print(input_data)

    # Transform the input data using the ColumnTransformer
    input_data_transformed = ct.transform(input_data).astype(float)
    #print(input_data_transformed)

    # Use the model to predict the probability of a collision
    prediction = model.predict(input_data_transformed)
    #print(prediction)

    return prediction[0][0] > 0.5


# ------------------------------------------------------------------------------
# TESTS

# Test model with examples
print(predict_collision("ASTEROID", 100, 100))  # Expected output: True
print(predict_collision("ASTEROID", 0, 100))    # Expected output: False
print(predict_collision("ASTEROID", 100, 0))    # Expected output: False
print(predict_collision("PARTICLE", 100, 100))  # Expected output: False
print(predict_collision("PARTICLE", 0, 100))    # Expected output: False
print(predict_collision("PARTICLE", 100, 0))    # Expected output: False
print(predict_collision("ASTEROID", 0, 0))    # Expected output: True
