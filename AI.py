import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
            data.append(re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', sentence.lower().strip("\n")))
        y += 1
        if y == 250:
            break

adjusted_data = []

keywords = ["kunskap i", "erfarenhet av", "skicklig på", "tränad i", "bekant med",
            "kompetent inom", "expertis inom", "kvalificerad i", "certifierad i",
            "behörig inom", "effektiv på", "färdig med", "sakkunnig i", "utbildad i",
            "specialiserad på", "proficient in", "knowledge of", "skilled in", 
            "experienced with", "trained in", "familiar with", "competent in", 
            "expertise in", "qualified in", "capable of", "adept in", "accomplished in",
            "specialized in", "certified in", "efficient in", "skills", "skills in"]

for row in data:
    if any(keyword in row for keyword in keywords):
        row *= 3
    
    adjusted_data.append(row)

vectorizer = TfidfVectorizer(stop_words=all_stopwords, token_pattern=r'\b[a-zA-ZåäöÅÄÖ]+\b', ngram_range=(1, 3), max_df=0.05, norm='l2')

x = vectorizer.fit_transform(adjusted_data)

feature_names = vectorizer.get_feature_names_out()

corpus = Sparse2Corpus(x, documents_columns=False)

id2word = {i: word for i, word in enumerate(feature_names)}

lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=250, id2word=id2word, passes=10, alpha=0.05, eta=0.8, random_state=42)

theText = open("data/text.txt", 'a', encoding="utf-8")

topics = lda_model.print_topics(num_topics=250, num_words=10)
for topic in topics:
    theText.write(str(re.sub(r'\d*\.\d+|\d+', '', topic[1]).replace('*', '').replace('  ', '').replace('"', '').strip()) + "\n")
