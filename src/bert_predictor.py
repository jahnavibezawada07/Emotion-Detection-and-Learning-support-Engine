import os
import pickle
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class BERTEmotionClassifier:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.emotion_labels = None

    ############################################################
    # LOAD MODEL
    ############################################################

    def load_model(self, model_path="models/BERT"):

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading BERT model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)

        ##self.model.eval()

        ############################################################
        # Load label mappings
        ############################################################
        
        label_path = os.path.join(model_path, "label_encoder.pkl")

        if os.path.exists(label_path):
            with open(label_path, "rb") as f:
                label_encoder = pickle.load(f)
#######needs to take assistance
            self.label_encoder = None
            self.label_encoder = label_encoder

        # Create id -> label mapping
            self.id2label = {
                i: label
                for i, label in enumerate(label_encoder.classes_)
            }

            # Store list of labels
            self.emotion_labels = list(label_encoder.classes_)

        
        else:
            raise FileNotFoundError("label_encoder.pkl not found!")

        print("Model Loaded Successfully!")
        print("Classes:", self.emotion_labels)

    ############################################################
    # PREDICT
    ############################################################

    def predict(self, text):

        if self.model is None:

            raise ValueError(
                "Model not loaded. Call load_model() first."
            )

        ############################################################
        # Tokenize
        ############################################################

        inputs = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            padding="max_length",
            max_length=80

        )

        inputs = {

            k: v.to(self.device)

            for k, v in inputs.items()

        }

        ############################################################
        # Run BERT
        ############################################################

        with torch.no_grad():

            outputs = self.model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()[0]

        ############################################################
        # CLASS WEIGHTING
        ############################################################

        class_weights = np.array([

            1.2,   # Bored

            1.8,   # Confident

            0.6,   # Confused

            1.0,   # Curious

            1.4    # Frustrated

        ])

        ############################################################
        # KEYWORD BOOSTING
        ############################################################

        text_lower = text.lower()

        confidence_keywords = [

            "comfortable",

            "confident",

            "easy",

            "clear",

            "understand",

            "understood",

            "got it",

            "makes sense",

            "simple"

        ]

        confusion_keywords = [

            "confused",

            "unclear",

            "lost",

            "don't understand",

            "cannot understand",

            "puzzled",

            "stuck"

        ]

        ############################################################
        # Boost Confidence
        ############################################################

        if any(
            keyword in text_lower
            for keyword in confidence_keywords
        ):

            class_weights[1] *= 2.5

            class_weights[2] *= 0.3

        ############################################################
        # Boost Confusion
        ############################################################

        elif any(
            keyword in text_lower
            for keyword in confusion_keywords
        ):

            class_weights[2] *= 2.0

        ############################################################
        # Apply weights
        ############################################################

        weighted_probs = probs * class_weights

        weighted_probs = weighted_probs / weighted_probs.sum()

        pred_id = np.argmax(weighted_probs)

        emotion = self.id2label[pred_id]

        ############################################################
        # Return prediction
        ############################################################

        return {

            "emotion": emotion,

            "confidence": float(weighted_probs[pred_id]),

            "scores": {

                self.id2label[i]:

                    float(weighted_probs[i])

                for i in range(len(weighted_probs))

            },

            "cleaned_text": text.strip()

        }
    