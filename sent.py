import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords

data_frame = pd.read_csv('twitter_data.csv', encoding = 'latin1')

stop_words_nl = set(stopwords.words('dutch'))
stop_words_en = set(stopwords.words('english'))
stop_words = stop_words_nl.union(stop_words_en)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]+', '', text)
    words = text.split()

    filtered_words = [word for word in words if word not in stop_words]
    clean_text = " ".join(filtered_words)

    return clean_text

#data_frame['clean_text'] = data_frame['text_column'].apply(clean_text)
#print(data_frame[['text_column', 'clean_text']].head())
#print(data_frame['clean_text'].isnull().sum())

