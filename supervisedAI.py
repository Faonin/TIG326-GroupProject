import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load data
data = pd.read_csv('data/training.csv')  # Ensure you have a CSV file with 'Word' and 'Category'

# Text preprocessing
tokenizer = Tokenizer(num_words=10000)  # Adjust based on your vocabulary size
tokenizer.fit_on_texts(data['word'])
sequences = tokenizer.texts_to_sequences(data['word'])

# Determine max length for padding
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encoding labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['category'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded, test_size=0.2, random_state=42)

# Building the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=max_length))  # Adjust embedding layer parameters as needed
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # The output layer nodes equal the number of categories

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_split=0.4)

new_data_set = set()

# Read incoming info and gather unique entries
with open("data/text.txt", encoding="utf-8") as incoming_info:
    for line in incoming_info:
        # Split line into phrases, remove duplicates in the line
        phrases = set(line.strip().split(" + "))
        # Add processed phrases to the set of unique entries
        new_data_set.update(phrases)

new_sequences = tokenizer.texts_to_sequences(new_data_set)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')

predictions = model.predict(new_padded_sequences)

predicted_categories = np.argmax(predictions, axis=1)

predicted_category_names = label_encoder.inverse_transform(predicted_categories)

with open("data/predicted_data.csv", "w", encoding="utf-8") as file:
    file.write("Phrase, Predicted Category\n")
    for phrase, category in zip(new_data_set, predicted_category_names):
        file.write(f"{phrase},{category}\n")