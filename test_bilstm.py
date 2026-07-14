from src.emotion_predictor import EmotionPredictor

predictor = EmotionPredictor()

print(predictor.tokenizer.texts_to_sequences(
    ["finally understood recursion today"]
))

result = predictor.predict(
    "I finally understood recursion today!"
)

print(result)

tests = [
    "I am completely confused by this chapter.",
    "I finally solved the assignment!",
    "This lecture is so boring.",
    "I want to know how this algorithm works.",
    "I have tried everything and I'm still stuck."
]

for t in tests:
    print(t)
    print(predictor.predict(t))
    print("-" * 50)