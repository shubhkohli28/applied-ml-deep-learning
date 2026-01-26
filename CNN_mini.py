import os
import time
import sys

import tensorflow as tf
from tensorflow import keras
datasets = keras.datasets
layers = keras.layers
models = keras.models

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Prefer tensorflow.keras, fall back to standalone keras if tensorflow's subpackage isn't resolvable
try:
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import ImageDataGenerator


# -----------------------------
# 1. Load and preprocess CIFAR-10
# -----------------------------
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# OPTIONAL: use a smaller subset to speed up on CPU (uncomment if still too slow)
# x_train = x_train[:20000]
# y_train = y_train[:20000]

# Train/validation split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten labels
y_train = y_train.flatten()
y_val   = y_val.flatten()
y_test  = y_test.flatten()


# -----------------------------
# 2. Data generators
# -----------------------------
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Simple generators for validation/test (NO augmentation)
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_datagen.fit(x_train)

batch_size = 64

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator   = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
test_generator  = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)


# -----------------------------
# 3. Build Transfer Learning Model (VGG16)
# -----------------------------
base_model = VGG16(
    weights='imagenet',          # make sure spelling is correct
    include_top=False,
    input_shape=(32, 32, 3)
)
base_model.trainable = False     # freeze feature extractor

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# -----------------------------
# 4. Train the model (made faster)
# -----------------------------
# Main speedups:
# - epochs reduced from 20 to 5
# - you can also use fewer steps_per_epoch if needed

epochs = 5  # was 20; change this if you want more training later

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    # You can cap steps_per_epoch and validation_steps to speed up further:
    # steps_per_epoch=300,         # instead of len(train_generator)
    # validation_steps=100,
    verbose=0
)

# -----------------------------
# 5. Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f'Test accuracy: {test_acc:.4f}')


# -----------------------------
# 6. Plot training curves
# -----------------------------
# Accuracy
plt.figure()
plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Loss
plt.figure()
plt.plot(history.history.get('loss', []), label='Train Loss')
plt.plot(history.history.get('val_loss', []), label='Val Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
