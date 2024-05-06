import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import gensim
from gensim.matutils import Sparse2Corpus
import re
import os

keywords = ["kunskap i", "erfarenhet av", "skicklig på", "tränad i", "bekant med",
            "kompetent inom", "expertis inom", "kvalificerad i", "certifierad i",
            "behörig inom", "effektiv på", "färdig med", "sakkunnig i", "utbildad i",
            "specialiserad på", "proficient in", "knowledge of", "skilled in", "requirements"
            "experienced with", "trained in", "familiar with", "competent in", 
            "expertise in", "qualified in", "capable of", "adept in", "accomplished in",
            "specialized in", "certified in", "efficient in", "skills", "skills in", "krav", "competencies"]

num_of_work_categories = 3000

nltk.download('stopwords')

english_stopwords = stopwords.words('english')
swedish_stopwords = stopwords.words('swedish')

all_stopwords = english_stopwords + swedish_stopwords

num_lines = sum(1 for _ in open('data/data.json'))

with open("data/data.json", 'r', encoding="utf-8") as dataset:
    data = []
    y = 0
    for i in dataset:
        i = json.loads(i)
        for sentence in i["description"]["text"].split("."):
            data.append(re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', sentence.lower().strip("\n")))
        y += 1
        if y % 100000 == 0:
            print(str(round((y / num_lines) * 100, 2)) + "%")

adjusted_data = []

for row in data:
    if any(keyword in row for keyword in keywords):
        adjusted_data.append(row)

vectorizer = CountVectorizer(stop_words=all_stopwords, token_pattern=r'\b[a-zA-ZåäöÅÄÖ]+\b', ngram_range=(1, 3))

vector_document = vectorizer.fit_transform(adjusted_data)

feature_names = vectorizer.get_feature_names_out()

corpus = Sparse2Corpus(vector_document, documents_columns=False)

id2word = {i: word for i, word in enumerate(feature_names)}

gensim.corpora.MmCorpus.serialize('./data/vector.mm', corpus)

mm_corpus = gensim.corpora.MmCorpus('./data/vector.mm')

print("AI running please be patient it might take a few hours")

lda_model = gensim.models.LdaModel(corpus=mm_corpus, id2word=id2word, num_topics=num_of_work_categories, decay=1, passes=5, alpha=0.02, eta=5, random_state=42)

theText = open("data/text.txt", 'a', encoding="utf-8")

topics = lda_model.print_topics(num_words=100, num_topics=num_of_work_categories)

for topic in topics:
    theText.write(str(re.sub(r'\d*\.\d+|\d+', '', topic[1]).replace('*', '').replace('  ', '').replace('"', '').strip()) + "\n")

os.remove("data/vector.mm")
os.remove("data/vector.mm.index")