from keras.models import load_model
from src import pre_process_util as ppu
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np

model = load_model('resources/noticias_g1_modelo_rede_neural.h5')
categories = pd.read_csv('resources/noticias_g1_categorias.csv')['categoria']


def classify_news(news):
    news = ppu.prepare_news_for_neural(news)
    df_news = pd.DataFrame.sparse.from_spmatrix(news)
    sm_news = coo_matrix(df_news)
    predicted = model.predict(sm_news)[0]
    return predicted, categories[np.argmax(predicted)]
