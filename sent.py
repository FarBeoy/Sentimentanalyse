import pandas as pd 
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#--- Reads data ---#
data_frame = pd.read_csv('twitter_data.csv', encoding = 'latin1', header = None)
data_frame.columns = ['id', 'time', 'date', 'query', 'username', 'text']

stop_words = set(stopwords.words('english'))

#--- Use if stopwords import does not work ---#
#def load_stopwords(file_path):
    #with open(file_path, 'r') as file:
        #return set(line.strip() for line in file)

#stop_words = load_stopwords('stopwords.txt')

#--- Cleaning text ---#
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]+', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    clean_text = " ".join(filtered_words)
    return clean_text

data_frame['clean_text'] = data_frame['text'].apply(clean_text)
print(data_frame[['text', 'clean_text']].head())
print(data_frame['clean_text'].isnull().sum())

#--- Applies sentiment to cleaned text ---#
sia_obj = SentimentIntensityAnalyzer()

def sia_scores(sentence):
    sia_dict = sia_obj.polarity_scores(sentence)
    return sia_dict['compound']

data_frame['sentiment_score'] = data_frame['clean_text'].apply(sia_scores)
print(data_frame[['text', 'clean_text', 'sentiment_score']].head())

#--- User input analysis ---#
print("\nEnter a text for analysis: ")
user_input = input("Text: ")
sentiment_dict = sia_obj.polarity_scores(user_input)

print("Overall sentiment dictionary is: ", sentiment_dict)
print("Sentence was rated as: ", sentiment_dict['neg'] * 100, "% Negative")
print("Sentence was rated as: ", sentiment_dict['neu'] * 100, "% Neutral")
print("Sentence was rated as: ", sentiment_dict['pos'] * 100, "% Positive")
print("Sentence overall rated as: ", end = "")

if sentiment_dict['compound'] >= 0.05:
    print("Positive")
elif sentiment_dict['compound'] <= -0.05:
    print("Negative")
else:
    print("Neutral")
