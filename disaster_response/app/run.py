import json
import plotly
import pandas as pd
# import joblib
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
model = pickle.load(open("../models/classifier.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = []
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one = [Bar(
        x=genre_names,
        y=genre_counts)]

    layout_one = dict(
        title='Distribution of Message Genres',
        y_axis=dict(title="Count"),
        x_axis=dict(title='Genre')
    )

    # Graph 2 to display distribution of different categories
    categories = df.iloc[:, 4:]
    categories_sum = categories.sum(axis=0).sort_values(ascending=False)
    categories_names = list(categories_sum.index)

    graph_two = [Bar(
        x=categories_names,
        y=categories_sum)]

    layout_two = dict(
        title='Distribution of different categories',
        y_axis=dict(title="Count"),
        x_axis=dict(title='Category')
    )

    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #     app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(port=8080)


if __name__ == '__main__':
    main()
