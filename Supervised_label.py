"""import json
import sys
import csv

# Configure stdout and stderr to handle UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def stream_job_descriptions(file_path):
   
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                job_ad = json.loads(line)
                if 'description' in job_ad and 'text' in job_ad['description']:
                    # Replace any pipe characters in the description to avoid confusion with the delimiter
                    text = job_ad['description']['text'].replace('|', ' ')
                    yield text
            except json.JSONDecodeError:
                print("Error decoding JSON", file=sys.stderr)
            except Exception as e:
                print(f"An error occurred: {e}", file=sys.stderr)

def save_descriptions_to_csv(file_path, output_file):
   
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # Set the delimiter to a pipe character
        writer = csv.writer(f, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Description'])
        for description in stream_job_descriptions(file_path):
            writer.writerow([description])

if __name__ == "__main__":
    file_path = 'data/2022.jsonl'
    output_file = 'data/job_descriptions.csv'
    save_descriptions_to_csv(file_path, output_file)"""


import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.stats import uniform

# Load the Swedish language model for spaCy with certain components disabled for efficiency
nlp = spacy.load('sv_core_news_sm', disable=['parser', 'ner'])

# Ensure that nltk has the necessary resources
nltk.download('stopwords')

# Load Swedish stopwords from both NLTK and spaCy, and convert the set to a list for compatibility
swedish_stopwords = list(set(stopwords.words('swedish')) | set(nlp.Defaults.stop_words))

# Function to dynamically label data (simple example)
def label_data(description):
    keywords = {
        'Management': ['chef', 'ledare', 'verksamhetsledare', 'management', 'direktör'],
        'IT': ['utvecklare', 'programmerare', 'system', 'it', 'programvara', 'nätverk', 'cybersäkerhet', 'teknisk support'],
        'Sales': ['försäljning', 'kund', 'säljare', 'sales', 'marknadsföring', 'butikschef'],
        'Healthcare': ['vårdbiträde', 'sjuksköterska', 'läkare', 'vård', 'medicinsk', 'hälso-', 'sjukvård']
    }
    
    doc = nlp(description.lower())
    for label, key_list in keywords.items():
        if any(key.text in key_list for key in doc):
            return label
    return 'Other'

# Read the dataset
data = pd.read_csv('data/job_descriptions.csv', delimiter='|')

# Sample a random subset of data for development and testing
sampled_data = data.sample(frac=0.1)  # Adjust fraction according to your needs, e.g., 10% of the data

sampled_data['Description'] = sampled_data['Description'].fillna('')  # Replace NaNs with empty strings
sampled_data['label'] = sampled_data['Description'].apply(label_data)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sampled_data['Description'], sampled_data['label'], test_size=0.3, random_state=42)

# Prepare the preprocessing and modeling pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=swedish_stopwords)),
    ('classifier', SGDClassifier(random_state=42))
])

# Use RandomizedSearchCV for parameter selection
param_dist = {
    'tfidf__max_df': uniform(0.5, 0.5),
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'classifier__alpha': uniform(0.0001, 0.01)
}
search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
search.fit(X_train, y_train)

# Prediction and evaluation using the best estimator
predictions = search.predict(X_test)
print(classification_report(y_test, predictions))

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Description': X_test,
    'Actual Label': y_test,
    'Predicted Label': predictions
})

# Export the DataFrame to a CSV file
results_df.to_csv('data/model_results.csv', index=False)



