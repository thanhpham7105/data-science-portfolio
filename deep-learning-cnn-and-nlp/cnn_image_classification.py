# ============================================================
# Seedling Image Classification with a Convolutional Neural Network
# ============================================================

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Part B – Data Preparation
# ============================================================

# ---------------------------
# B1 – Load images and labels
# ---------------------------
images = np.load("images.npy")        # expected shape: (4750, 128, 128, 3)
labels_df = pd.read_csv("labels.csv")

print("Images shape:", images.shape)
print("Labels shape:", labels_df.shape)
print(labels_df.head())

# Ensure label column name
if "label" not in labels_df.columns:
    labels_df = labels_df.rename(columns={labels_df.columns[0]: "label"})

# ---------------------------
# B1a – Class distribution
# ---------------------------
plt.figure(figsize=(10, 4))
labels_df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution of Seedling Species")
plt.xlabel("Species")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()   # Screenshot for B1a

# ---------------------------
# B1b – Sample images with labels
# ---------------------------
plt.figure(figsize=(12, 6))
num_samples = 12
for i in range(num_samples):
    idx = np.random.randint(0, len(images))
    plt.subplot(3, 4, i+1)
    plt.imshow(images[idx].astype("uint8"))
    plt.title(labels_df.iloc[idx]["label"])
    plt.axis("off")
plt.tight_layout()
plt.show()   # Screenshot for B1b

# ---------------------------
# B2 – Data augmentation layer
# ---------------------------
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)

# ---------------------------
# B3 – Normalization
# ---------------------------
images = images.astype("float32") / 255.0
print("Pixel range after normalization:", images.min(), "to", images.max())

# ---------------------------
# B4 – Train/Validation/Test split
# ---------------------------
y = labels_df["label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    y,
    test_size=0.30,          # 30% for val+test
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,          # 15% val, 15% test
    stratify=y_temp,
    random_state=42
)

print("Train set:", X_train.shape, "Labels:", len(y_train))
print("Val set:  ", X_val.shape,   "Labels:", len(y_val))
print("Test set: ", X_test.shape,  "Labels:", len(y_test))

# ---------------------------
# B5 – Encode target labels
# ---------------------------
label_encoder = LabelEncoder()
y_train_int = label_encoder.fit_transform(y_train)
y_val_int   = label_encoder.transform(y_val)
y_test_int  = label_encoder.transform(y_test)

class_names = label_encoder.classes_
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# ---------------------------
# B6 – Save prepared datasets
# ---------------------------
np.save("X_train.npy", X_train)
np.save("X_val.npy",   X_val)
np.save("X_test.npy",  X_test)
np.save("y_train_int.npy", y_train_int)
np.save("y_val_int.npy",   y_val_int)
np.save("y_test_int.npy",  y_test_int)

print("Prepared datasets saved (X_*.npy and y_*.npy).")

# ============================================================
# Part E – Network Architecture
# ============================================================

# ---------------------------
# Build CNN model
# ---------------------------
input_shape = (128, 128, 3)

model = models.Sequential(name="Seedling_CNN")
model.add(layers.Input(shape=input_shape))
model.add(data_augmentation)

model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()   # Screenshot for E1

# ---------------------------
# Compile model
# ---------------------------
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks: early stopping and checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_seedling_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ============================================================
# Part F – Model Training and Evaluation
# ============================================================

# ---------------------------
# F1a – Train model
# ---------------------------
history = model.fit(
    X_train, y_train_int,
    validation_data=(X_val, y_val_int),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# ---------------------------
# F1b/F1c – Plot loss and accuracy
# ---------------------------
train_loss = history.history["loss"]
val_loss   = history.history["val_loss"]
train_acc  = history.history["accuracy"]
val_acc    = history.history["val_accuracy"]

epochs_ran = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, train_loss, label="Training Loss")
plt.plot(epochs_ran, val_loss,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()   # Screenshot for F1c

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, train_acc, label="Training Accuracy")
plt.plot(epochs_ran, val_acc,   label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# F3 – Evaluate on test set
# ---------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test_int, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ---------------------------
# E4/F2/F3 – Confusion matrix
# ---------------------------
y_test_pred_probs = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

cm = confusion_matrix(y_test_int, y_test_pred)

print("Classification Report:")
print(classification_report(y_test_int, y_test_pred, target_names=class_names))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Seedling CNN")
plt.tight_layout()
plt.show()   # Screenshot for E4

# ============================================================
# Part G/H – Save Trained Model
# ============================================================
model.save("seedling_cnn_model.keras")
print("Final model saved as 'seedling_cnn_model.keras'")
