import re

def preprocess_text(text):
    """
    Basic text preprocessing for emotion detection.
    """
    text = text.lower().strip()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text

from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 80   # Use the same value you used during training

def preprocess_bilstm(text, tokenizer):
    text = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded

def preprocess_bert(text, tokenizer):
    text = preprocess_text(text)

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=80,
        return_tensors="pt"
    )

    return encoded
