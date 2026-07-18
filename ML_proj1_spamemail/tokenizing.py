import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split

# Load data from the previous step
balanced_data = pd.read_csv("cleaned_emails.csv") 
balanced_data['text'] = balanced_data['text'].fillna('') #fills all the blank or null entries with a empty space to avoid errors

## Tokenization & Padding

train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
)
#the above tells the program how to split the test data- balanced_data['text'] is the actual email content whilst the 'label' is the spam or ham assigned to the email
#train is what the program will train on and test is what's held back to trial the model later
#the test_size of 0.2 represents 20% of all the data values will be used to test the rest will be used in training

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(train_X) #scans data sets and assigns values to all the words with 1 assigned to most frequently used etc..

train_sequences = tokenizer.texts_to_sequences(train_X) #swaps every word for the assigned value or token
test_sequences = tokenizer.texts_to_sequences(test_X) #swaps every word for the assigned value or token

max_len = 100  # Maximum sequence length retained for both
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

#adds 0's for every message below 100 and cuts off messages above 100 in length

train_Y = (train_Y == 'spam').astype(int) 
test_Y = (test_Y == 'spam').astype(int)

np.savez("processed_data.npz", 
         train_X=train_sequences, 
         test_X=test_sequences, 
         train_Y=train_Y, 
         test_Y=test_Y,
         vocab_size=np.array([len(tokenizer.word_index) + 1])) #bundles all arrays together in a npz file and saves tokenized sequences and the text labels
#it also counts how many unique words were found as tokens

print("File 2 Complete! Array matrices saved to 'processed_data.npz'")