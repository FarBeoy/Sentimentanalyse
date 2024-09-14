from flask import Flask, render_template, request
import pandas as pd 
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#--- Reads data ---#
app = Flask(__name__)

data_frame = pd.read_csv('twitter_data.csv', encoding = 'latin1', header = None)
data_frame.columns = ['id', 'time', 'date', 'query', 'username', 'text']

stop_words = set(stopwords.words('english'))

#--- Use if nltk/stopwords import does not work ---#
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

#--- Applies sentiment to cleaned text ---#
sia_obj = SentimentIntensityAnalyzer()

def sia_scores(sentence):
    sia_dict = sia_obj.polarity_scores(sentence)
    return sia_dict['compound']

data_frame['sentiment_score'] = data_frame['clean_text'].apply(sia_scores)

#--- Flask render ---#
@app.route('/')
def index():
    return render_template('index.html')

#--- User input analysis ---#
@app.route('/analyze', methods = ['POST'])
def analyze():
    if request.method == 'POST':
        user_input = request.form['user_input']

        clean_input = clean_text(user_input)
        sentiment_dict = sia_obj.polarity_scores(user_input)

        if sentiment_dict['compound'] >= 0.05:
            overall_sentiment = "Positive"
        elif sentiment_dict['compound'] <= -0.05:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        return render_template('result.html',
                               original_text = user_input,
                               clean_text = clean_input,
                               neg = sentiment_dict['neg'] * 100,
                               neu = sentiment_dict['neu'] * 100,
                               pos = sentiment_dict['pos'] * 100,
                               overall = overall_sentiment)

if __name__ == '__main__':
    app.run(debug = True)
