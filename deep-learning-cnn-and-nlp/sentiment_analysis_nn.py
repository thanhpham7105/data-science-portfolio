# Sentiment Analysis with an Embedding-Based Neural Network
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data Preparation

# Load all 3 dataset files
amazon_df = pd.read_csv("amazon_cells_labelled.txt",
                        sep="\t", header=None, names=["sentence", "label"])

imdb_df = pd.read_csv("imdb_labelled.txt",
                      sep="\t", header=None, names=["sentence", "label"])

yelp_df = pd.read_csv("yelp_labelled.txt",
                      sep="\t", header=None, names=["sentence", "label"])

# Combine them into one dataset
data = pd.concat([amazon_df, imdb_df, yelp_df], ignore_index=True)

# Show results
print("Combined dataset shape:", data.shape)
print(data.head())
print("\nLabel counts:")
print(data["label"].value_counts())
print(data.head())

# B1 – Check for unusual characters

def has_non_ascii(text):
    return any(ord(c) > 127 for c in text)

data["has_non_ascii"] = data["sentence"].apply(has_non_ascii)
print("\nRows with non-ASCII characters:", data["has_non_ascii"].sum())

if data["has_non_ascii"].sum() > 0:
    print("\nExamples with non-ASCII characters:")
    print(data[data["has_non_ascii"]].head())

# ---------------------------
# B1 – Basic cleaning / normalization
# ---------------------------

def basic_clean(text):
    # Lowercase and normalize whitespace; you can extend with more steps if needed
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["clean_sentence"] = data["sentence"].apply(basic_clean)

print("\nOriginal vs cleaned examples:")
print(data[["sentence", "clean_sentence"]].head())

# B1 – EDA: vocabulary size & sequence lengths
#     (Initial tokenizer just for statistics)
eda_tokenizer = Tokenizer(oov_token="<OOV>")
eda_tokenizer.fit_on_texts(data["clean_sentence"])

word_index_full = eda_tokenizer.word_index
vocab_size_full = len(word_index_full) + 1  # +1 for padding index

print("\nFull vocabulary size (no cap yet):", vocab_size_full)

eda_sequences = eda_tokenizer.texts_to_sequences(data["clean_sentence"])
seq_lengths = [len(s) for s in eda_sequences]

print("Min sequence length:", int(np.min(seq_lengths)))
print("Max sequence length:", int(np.max(seq_lengths)))
print("Mean sequence length:", float(np.mean(seq_lengths)))

plt.figure()
plt.hist(seq_lengths, bins=30)
plt.xlabel("Sequence length (tokens)")
plt.ylabel("Count")
plt.title("Distribution of sentence lengths")
plt.show()

for p in [90, 95, 99]:
    print(f"{p}th percentile sequence length:", np.percentile(seq_lengths, p))

# ---------------------------
# B1 – Choose hyperparameters for text representation

vocab_size = 5000        # limit to the most frequent words
embedding_dim = 16       # small embedding size, common in basic text demos
max_len = 40             # based on length distribution (e.g., around 95th percentile)
trunc_type = "post"
pad_type = "post"

print("\nChosen vocab_size:", vocab_size)
print("Chosen embedding_dim:", embedding_dim)
print("Chosen max_len:", max_len)

# B2 – Final tokenization with cap on vocabulary
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data["clean_sentence"])
sequences = tokenizer.texts_to_sequences(data["clean_sentence"])

sample_idx = 0
print("\nExample tokenized sentence:")
print("Cleaned text:", data['clean_sentence'].iloc[sample_idx])
print("Tokenized:", sequences[sample_idx])

# B3 – Padding sequences
padded = pad_sequences(
    sequences,
    maxlen=max_len,
    padding=pad_type,
    truncating=trunc_type
)

labels = data["label"].values

print("\nPadded sequences shape:", padded.shape)
print("\nExample padded sequence (same sample):")
print("Padded:", padded[sample_idx])

# B4 – Categories of sentiment
# Two categories: 0 = negative, 1 = positive
num_classes = 2
print("\nNumber of sentiment classes:", num_classes)
print("Final layer will use sigmoid activation for binary classification.")

# B5 – Train / validation / test split
X_temp, X_test, y_temp, y_test = train_test_split(
    padded,
    labels,
    test_size=0.15,
    random_state=SEED,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,  # makes overall split ≈ 70/15/15
    random_state=SEED,
    stratify=y_temp
)

print("\nTrain size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

# B6 – Save prepared dataset
train_df = pd.DataFrame(X_train)
train_df["label"] = y_train
train_df.to_csv("prepared_train_data.csv", index=False)

val_df = pd.DataFrame(X_val)
val_df["label"] = y_val
val_df.to_csv("prepared_val_data.csv", index=False)

test_df = pd.DataFrame(X_test)
test_df["label"] = y_test
test_df.to_csv("prepared_test_data.csv", index=False)

print("\nPrepared datasets saved as CSV files.")
# Part C – Network Architecture (Text CNN)
# C1 & C2 – Define CNN model
model = models.Sequential([
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="embedding"
    ),
    layers.GlobalAveragePooling1D(name="global_avg_pool"),
    layers.Dense(16, activation="relu", name="dense1"),
    layers.Dense(1, activation="sigmoid", name="output")  # binary sentiment
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()
# Part D – Training and Evaluation
# D1 – Early stopping & training

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)
# D3 – Plot training curves
history_dict = history.history
epochs_ran = range(1, len(history_dict["loss"]) + 1)

# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(epochs_ran, history_dict["loss"], label="Train loss")
plt.plot(epochs_ran, history_dict["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(epochs_ran, history_dict["accuracy"], label="Train accuracy")
plt.plot(epochs_ran, history_dict["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# D2 & D4 – Test evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Predictions for more metrics
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype("int32").ravel()

print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion matrix (test set):")
print(confusion_matrix(y_test, y_pred))

# Part E – Save the model
model.save("sentiment_cnn_model.keras")
print("Model saved to sentiment_cnn_model.keras")
