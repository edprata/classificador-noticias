from src import pre_process_util as ppu
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('resources/noticias_g1_modelo_knn.pickle', 'rb'))
categories = pd.read_csv('resources/noticias_g1_categorias.csv')['categoria']


def classify_news(news):
    news = ppu.prepare_news(news, 250)
    predicted = model.predict(news)[0]
    return predicted, categories[np.argmax(predicted)]
