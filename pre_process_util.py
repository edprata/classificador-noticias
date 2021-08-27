from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import RSLPStemmer
import pickle
import nltk
import re

nltk.download('rslp')

with open('noticias_g1_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def clean_text(text):
    text = re.sub('[^a-zA-ZéúíóáÉÚÍÓÁèùìòàÈÙÌÒÀõãñÕÃÑêûîôâÊÛÎÔÂëÿüïöäËYÜÏÖÄçÇ\-\s]', '', text)
    text = re.sub('(\-..es)|(\-..e)|(\-.e)|(\-.os)|(\-ei)', '', text)
    text = re.sub('\-+', '', text)
    return text.lower()


def apply_stem(text):
    stemmer_ptbr = RSLPStemmer()
    return [stemmer_ptbr.stem(word) for word in text]


def prepare_news(news, news_lenght):
    hygienized = clean_text(news)
    stemmed = apply_stem(hygienized.split())
    vectorized = tokenizer.texts_to_sequences([stemmed])
    return pad_sequences(vectorized, padding='post', truncating='post', maxlen=news_lenght)
