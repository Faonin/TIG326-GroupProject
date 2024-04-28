import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import gensim
from gensim.matutils import Sparse2Corpus
import re

nltk.download('stopwords')

english_stopwords = stopwords.words('english')
swedish_stopwords = stopwords.words('swedish')

all_stopwords = english_stopwords + swedish_stopwords

with open("data/data.json", 'r', encoding="utf-8") as dataset:
    data = []
    y = 0
    for i in dataset:
        i = json.loads(i)
        for sentence in i["description"]["text"].split("."):
            data.append(sentence)
        y += 1
        if y == 150:
            break


vectorizer = CountVectorizer(stop_words=all_stopwords, token_pattern=r'\b[a-zA-ZåäöÅÄÖ]+\b', ngram_range=(1, 2))

x = vectorizer.fit_transform(data)

feature_names = vectorizer.get_feature_names_out()

# Convert sparse matrix to gensim corpus
corpus = Sparse2Corpus(x, documents_columns=False)

# Create id to word dictionary
id2word = {i: word for i, word in enumerate(feature_names)}

lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=150, id2word=id2word, passes=5, random_state=42)


theText = open("data/text.txt", 'a', encoding="utf-8")

# Print the topics identified by LDA
topics = lda_model.print_topics(num_topics=150, num_words=25)
for topic in topics:
    theText.write(str(re.sub(r'\d*\.\d+|\d+', '', topic[1]).replace('*', '').replace('  ', '').replace('"', '').strip()) + "\n")
