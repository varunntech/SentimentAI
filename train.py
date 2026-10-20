
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

np.random.seed(42)
tf.random.set_seed(42)

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 32
LSTM_UNITS = 32
EPOCHS = 3
BATCH_SIZE = 128

os.makedirs("models", exist_ok=True)

print("Loading IMDb dataset...")

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = Sequential([
    Input(shape=(MAX_LENGTH,), dtype="int32"),
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    LSTM(
        LSTM_UNITS
    ),
    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

print("\nEvaluating model...")

y_prob = model.predict(
    X_test,
    batch_size=BATCH_SIZE
).ravel()

y_pred = (y_prob >= 0.5).astype(int)

print("\nTest Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

model.save("models/sentiment_lstm.keras")

word_index = imdb.get_word_index()

np.savez(
    "models/imdb_word_index.npz",
    words=np.array(list(word_index.keys())),
    indices=np.array(list(word_index.values()))
)

print("\nModel and vocabulary saved successfully!")

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Training Performance")
plt.legend()
plt.tight_layout()
plt.savefig("models/training_accuracy.png")
plt.close()

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()
plt.title("LSTM Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("Evaluation plots saved in the models folder.")
