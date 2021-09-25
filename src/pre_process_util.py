from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import RSLPStemmer
import pickle
import nltk
import re

nltk.download('rslp')
nltk.download('stopwords')

stemmer_ptbr = RSLPStemmer()
stopwords = nltk.corpus.stopwords.words('portuguese')

tokenizer = pickle.load(open('resources/noticias_g1_tokenizer.pickle', 'rb'))
vectorizer_tfidf = pickle.load(open('resources/noticias_g1_vectorizer_tfidf.pickle', 'rb'))


def prepare_news_for_lstm(news, news_lenght):
    stemmed = prepare_text(news)
    vectorized = tokenizer.texts_to_sequences([stemmed])
    return pad_sequences(vectorized, padding='post', truncating='post', maxlen=news_lenght)


def prepare_news_for_neural(news):
    text = prepare_text(news)
    print(str(text))
    return vectorizer_tfidf.transform([text])


def prepare_text(news):
    hygienized = clean_text(news)
    without_stop_words = remove_stop_words(hygienized)
    stemed = apply_stem(without_stop_words)
    return " ".join(stemed)


def clean_text(text):
    text = re.sub('[^a-zA-ZéúíóáÉÚÍÓÁèùìòàÈÙÌÒÀõãñÕÃÑêûîôâÊÛÎÔÂëÿüïöäËYÜÏÖÄçÇ\-\s]', '', text)
    text = re.sub('(\-..es)|(\-..e)|(\-.e)|(\-.os)|(\-ei)', '', text)
    text = re.sub('\-+', '', text)
    return text.lower()


def remove_stop_words(text):
    return [word for word in text.split() if word not in stopwords]


def apply_stem(text):
    return [stemmer_ptbr.stem(word) for word in text]

