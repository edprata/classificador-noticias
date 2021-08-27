from keras.models import load_model
import pre_process_util as ppu
import pandas as pd
import numpy as np

model = load_model('noticias_g1_modelo_lstm.h5')
categories = pd.read_csv('noticias_g1_categorias.csv')['categoria']


def classify_news(news):
    news = ppu.prepare_news(news, 250)
    predicted = model.predict(news)[0]
    return predicted, categories[np.argmax(predicted)]
