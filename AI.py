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
            if len(sentence) != "\n":
                data.append(re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', sentence.lower()))
        y += 1
        if y == 300:
            break

keywords = ["kunskap i", "erfarenhet av", "skicklig på", "tränad i", "bekant med",
            "kompetent inom", "expertis inom", "kvalificerad i", "certifierad i",
            "behörig inom", "effektiv på", "färdig med", "sakkunnig i", "utbildad i",
            "specialiserad på", "supportera inom", "proficient in", "knowledge of", "skilled in", 
            "experienced with", "trained in", "familiar with", "competent in", 
            "expertise in", "qualified in", "capable of", "adept in", "accomplished in", "arbetsuppgifter"
            "specialized in", "certified in", "efficient in", "skills", "skills in", "meriterande", "Vi söker dig som har"]

adjusted_data = []
for doc in data:
    if any(keyword in doc.lower() for keyword in keywords):
        doc *= 150
    adjusted_data.append(doc)


vectorizer = CountVectorizer(stop_words=all_stopwords, token_pattern=r'\b[a-zA-ZåäöÅÄÖ]+\b', ngram_range=(1, 3))

x = vectorizer.fit_transform(adjusted_data)

feature_names = vectorizer.get_feature_names_out()

# Convert sparse matrix to gensim corpus
corpus = Sparse2Corpus(x, documents_columns=False)

# Create id to word dictionary
id2word = {i: word for i, word in enumerate(feature_names)}

lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=300, id2word=id2word, passes=25, random_state=42, iterations=5)

theText = open("data/text.txt", 'a', encoding="utf-8")

# Print the topics identified by LDA
topics = lda_model.print_topics(num_topics=300, num_words=50)
for topic in topics:
    theText.write(str(re.sub(r'\d*\.\d+|\d+', '', topic[1]).replace('*', '').replace('  ', '').replace('"', '').strip()) + "\n")
