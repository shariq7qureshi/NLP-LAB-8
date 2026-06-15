import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import joblib

# -------------------
# Load dataset
# -------------------

df = pd.read_csv("symptom_disease.csv")

texts = df["text"].astype(str)
labels = df["label"].astype(str)

# -------------------
# Encode labels
# -------------------

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

joblib.dump(encoder, "label_encoder.pkl")

# -------------------
# Split
# -------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------
# Vectorizer
# -------------------

max_tokens = 10000
sequence_length = 100

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorizer.adapt(X_train)

# save vocabulary

vocab = vectorizer.get_vocabulary()

joblib.dump(vocab, "vocab.pkl")

# -------------------
# Build model
# -------------------

model = Sequential([
    vectorizer,
    Embedding(max_tokens, 128),
    GlobalAveragePooling1D(),
    Dense(128, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test)
)

model.save("model.keras")

print("Training Complete!")
