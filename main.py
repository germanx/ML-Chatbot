import json
from sklearn.feature_extraction.text import CountVectorizer
import nltk   # NLTK.org - natural language tool kit
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
import re
import random
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters
# from telegram.ext import ApplicationBuilder


def filter_text(text):
  text = text.lower()
  pattern = r'[^\w\s]'
  text = re.sub(pattern, "", text)
  return text


# На вход: два текста, на выход: boolean(True, False)
# Функция isMatch вернет True, если тексты совпадают или False иначе
def is_match(text1, text2):
    text1 = filter_text(text1)
    text2 = filter_text(text2)

    if len(text1) == 0 or len(text2) == 0:
        return False

    # Проверить что одна фраза является частью другой
    # Text1 содержит text2
    if text1.find(text2) != -1:
        return True

    # Text2 содержит text1
    if text2.find(text1) != -1:
        return True

    # Расстояние Левенштейна (edit distance = расстояние редактирования)
    distance = nltk.edit_distance(text1, text2)  # Кол-во символов 0...Inf
    length = (len(text1) + len(text2)) / 2  # Средняя длина двух текстов
    score = distance / length  # 0...1

    return score < 0.6


config_file = open('content/big_bot_config.json', 'r')
BIG_CONFIG = json.load(config_file)

X = []  # фразы
y = []  # намерения

# Собираем фразы и интенты из BIG_CONFIG в X,y
for name, intent in BIG_CONFIG["intents"].items():
  for example in intent["examples"]:
    X.append(example)
    y.append(name)
  for example in intent["responses"]:
    X.append(example)
    y.append(name)

print(len(X))

vectorizer = CountVectorizer()
vectorizer.fit(X)  # learning

# fastest selected model
model = RandomForestClassifier()
# model = LogisticRegression()
# model = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,100,100,))
vecX = vectorizer.transform(X)
model.fit(vecX, y)  # learn model
c = model.score(vecX, y)
print(c)


def get_intent_ml(text):
    vec_text = vectorizer.transform([text])
    intent = model.predict(vec_text)[0]
    return intent


def get_intent(text):
    for name, intent in BIG_CONFIG["intents"].items():
        for example in intent["examples"]:
            if is_match(text, example):
                return name
    return None


def bot(phrase):
    phrase = filter_text(phrase)
    # Напрямую найти ответ
    intent = get_intent(phrase)

    # Если напрямую интент не нашелся
    if not intent:
        # Попросить модель найти ответ
        intent = get_intent_ml(phrase)

    # Если намерение определено
    if intent:
        responses = BIG_CONFIG["intents"][intent]["responses"]  # Находим варианты ответов
        return random.choice(responses)

    failure = BIG_CONFIG["failure_phrases"]
    return random.choice(failure)
    # Выдать Failure Phrase


def bot_telegram_reply(update: Update, ctx):
    text = update.message.text
    reply = bot(text)
    name = update.message.chat.full_name
    update.message.reply_text(reply)
    print(f'[{name}] {text}: {reply}')


print('start bot')
BOT_KEY = 'XXXXX'
upd = Updater(BOT_KEY)
# upd = ApplicationBuilder().token(BOT_KEY).build()

handler = MessageHandler(Filters.text, bot_telegram_reply)
upd.dispatcher.add_handler(handler)
upd.start_polling()
upd.idle()
