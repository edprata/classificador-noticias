from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from src import pre_process_util as ppu
import pandas as pd
import numpy as np

model = load_model('resources/noticias_g1_modelo_lstm.h5')
categories = pd.read_csv('resources/noticias_g1_categorias.csv')['categoria']


def classify_news(news):
    news = ppu.prepare_news_for_lstm(news, 250)
    predicted = model.predict(news)[0]
    return predicted, categories[np.argmax(predicted)]


def hanking_words(news):
    news = ppu.prepare_news_for_lstm(news, 250)
    for word in news[0]:
        if word == 0: break
        seq = pad_sequences([[word]], padding='post', truncating='post', maxlen=250)
        predicted = model.predict(seq)[0]
        print(predicted, categories[np.argmax(predicted)])
    return "", ""