import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string 
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/Users/Tricia/Desktop/ML_proj1_spamemail/spam_ham_dataset.csv')
#print(data.head())
#print(data.shape)

sns.countplot(x='label', data=data)
plt.show()

## Balancing the Dataset

ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

# Downsample Ham emails to match the number of Spam emails so ML trains on equal data 
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)

# Combine balanced data
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

# Visualize the balanced dataset
sns.countplot(x='label', data=balanced_data)
plt.title("Balanced Distribution of Spam and Ham Emails")
plt.xticks(ticks=[0, 1], labels=['Ham (Not Spam)', 'Spam'])
plt.show()

#Cleaning the text

balanced_data['text'] = balanced_data['text'].str.replace('Subject', '') #replaces all the 'subjects' and then converts everything in the column to a text data type
print(balanced_data.head())

punctuations_list = string.punctuation #grabs the list of punctuation premade in python
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list) #deletes all the punctuation and makes a translation map (faster than regular function loop scans list only once)
    return text.translate(temp) #applies this deletion map to all the parameters run in the function

balanced_data['text']= balanced_data['text'].apply(lambda x: remove_punctuations(x)) # this takes the email removes the punctuation and saves it into the original column

print(balanced_data.head())

def remove_stopwords(text):
    stop_words = stopwords.words('english') #grabs list of english filler words/'stopwords' predefined in NLTK

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower() #converts all the words the lowercase so it matches NLTK's predefined list (it's case sensitive)

        if word not in stop_words:
            imp_words.append(word) #adds the list of important words to the list 

    output = " ".join(imp_words) #creates one sentence of all the important words without the filler words

    return output


balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text)) #applies the changes of stopwords removal into the file
print(balanced_data.head())


## Making a Word Cloud 

def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text']) #joins all the emails in the list together 
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus) #formats the wordcloud
    plt.figure(figsize=(7, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} Emails', fontsize=15) #plots the wordcloud
    plt.axis('off')
    plt.show()

plot_word_cloud(balanced_data[balanced_data['label'] == 'ham'], typ='Non-Spam') #generates the wordcloud for only those emails that aren't spam
plot_word_cloud(balanced_data[balanced_data['label'] == 'spam'], typ='Spam') #generates the wordcloud for emails that are spam

#Storing the Changes

balanced_data.to_csv("cleaned_emails.csv", index=False) #converts the new data into a new CSV file that does not have index numbers listed
print("File 1 Complete! Clean text saved to 'cleaned_emails.csv'")





