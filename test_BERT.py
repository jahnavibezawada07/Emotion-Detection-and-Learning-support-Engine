from src.bert_predictor import BERTEmotionClassifier

bert = BERTEmotionClassifier()

bert.load_model("models/BERT")
'''
#confident
print(bert.predict("I completely understand recursion now."))
print(bert.predict("I finally understand pointers after practicing all weekend."))
print(bert.predict("The professor explained dynamic programming really well. It finally makes sense."))
print(bert.predict("I solved every coding problem without looking at the solutions."))

#confused
print(bert.predict("I have watched five tutorials but I still don't understand recursion."))
print(bert.predict("The lecture completely went over my head."))
print(bert.predict("I have absolutely no idea how binary trees work."))
print(bert.predict("I don't understand the concept of inheritance in OOP."))
print(bert.predict("This is too difficult for me to grasp."))
print(bert.predict("I'm lost and don't know what to do next."))

#frustrated
print(bert.predict("I have been debugging this error for six hours and nothing works."))
print(bert.predict("I have been struggling with this problem for days."))
print(bert.predict("Every time I fix one bug another one appears."))
print(bert.predict("I'm sick of writing this assignment."))

#confused
print(bert.predict("I wonder how transformers actually understand language."))
print(bert.predict("I'd love to learn more about reinforcement learning."))
print(bert.predict("Can someone explain how attention works inside BERT?"))
'''

print(bert.predict("I finally understand recursion, but debugging my program is still frustrating."))

print(bert.predict("I don't understand today's lecture, but I'm excited to learn it tomorrow."))

print(bert.predict("The assignment is difficult, but I think I can finish it."))

print(bert.predict("I understand everything except dynamic programming."))

print(bert.predict("The lecture was boring, but the lab session was really interesting."))

print(bert.predict("I'm confused about the syntax, but I'm determined to master Python."))

print(bert.predict("I failed the quiz again. I'm so disappointed."))

print(bert.predict("bro recursion is frying my brain 😭"))

print(bert.predict("finally got this assignment done!!"))

print(bert.predict("why is this compiler yelling at me"))

print(bert.predict("i think i understand arrays now"))

print(bert.predict("this lecture was sooo boring"))

print(bert.predict("can anyone explain pointers in simple words"))

print(bert.predict("let's try building a chatbot next"))