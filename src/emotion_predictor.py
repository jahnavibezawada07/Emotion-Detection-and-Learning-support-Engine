import re
import numpy as np
import tensorflow as tf
import pickle

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 80

class EmotionPredictor:

    def __init__(
        self,
        model_path="models/BiLSTM/emotion_model.keras",
        tokenizer_path="models/BiLSTM/tokenizer.pkl",
        classes_path="models/BiLSTM/label_encoder.pkl"
    ):

        try:
            self.model = tf.keras.models.load_model(model_path)

        except Exception:
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

        with open(tokenizer_path, "rb") as f:
             self.tokenizer = pickle.load(f)

        with open(classes_path, "rb") as f:
             self.label_encoder = pickle.load(f)

        self.classes = list(self.label_encoder.classes_)

        self.stopwords = set(stopwords.words("english"))
        print("EmotionPredictor initialized successfully.")
        print("Classes:", self.classes)

    def clean_text(self, text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"[^a-zA-Z\s]", "", text)

        words = text.split()

        words = [
            word
            for word in words
            if word not in self.stopwords
        ]

        return " ".join(words)
    
    def predict(self, text):

        cleaned = self.clean_text(text)

        if cleaned.strip() == "":
            cleaned = text.lower()

        sequence = self.tokenizer.texts_to_sequences([cleaned])

        if len(sequence[0]) == 0:

            return {
                "emotion": "Confused",
                "confidence": 0.5,
                "scores": {
                    cls:0.2
                    for cls in self.classes
                },
                "cleaned_text": cleaned
            }

        padded = pad_sequences(
            sequence,
            maxlen=MAX_SEQ_LEN,
            padding="post",
            truncating="post"
        )

        probs = self.model.predict(
            padded,
            verbose=0
            ).flatten()
        
        prediction = int(np.argmax(probs, axis=-1))

        emotion = self.classes[prediction]

        confidence = float(probs[prediction])

        scores = {}

        for i, emotion_name in enumerate(self.classes):
            scores[emotion_name] = float(probs[i])

        return {

            "emotion": emotion,

            "confidence": confidence,

            "scores": scores,

            "cleaned_text": cleaned

        }
