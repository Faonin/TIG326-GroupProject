import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from preprocessing.text import Tokenizer
from preprocessing.sequence import pad_sequences


unique_entries = set()

# Read incoming info and gather unique entries
with open("data/text.txt", encoding="utf-8") as incoming_info:
    for line in incoming_info:
        # Split line into phrases, remove duplicates in the line
        phrases = set(line.strip().split(" + "))
        # Add processed phrases to the set of unique entries
        unique_entries.update(phrases)

# Write all unique entries to the output file
with open("data/all_code_words.csv", "w", encoding="utf-8") as output_file:
    for entry in unique_entries:
        output_file.write(f"{entry},övrigt\n")


# Load data
data = pd.read_csv('data/training.csv')  # Ensure you have a CSV file with 'Word' and 'Category'

# Text preprocessing
tokenizer = Tokenizer(num_words=1000)  # Adjust based on your vocabulary size
tokenizer.fit_on_texts(data['Word'])
sequences = tokenizer.texts_to_sequences(data['Word'])

# Determine max length for padding
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encoding labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Category'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded, test_size=0.2, random_state=42)

# Building the model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=50, input_length=max_length))  # Adjust embedding layer parameters as needed
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # The output layer nodes equal the number of categories

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)