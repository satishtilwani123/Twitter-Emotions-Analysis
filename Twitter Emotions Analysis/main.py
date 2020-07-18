import tkinter as tk
from tkinter import *
from tkinter import ttk
from ttkthemes import themed_tk as tk1
from PIL import ImageTk,Image
from tkinter import filedialog
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup
from sklearn import metrics
import pandas as pd
import numpy as np
import string
import json
import nltk
import csv
import re, urllib
from imblearn.over_sampling import SMOTE

stemming = PorterStemmer()
stops = set(stopwords.words("english"))
tokenizer = nltk.RegexpTokenizer(r"\w+")

def remove_urls(text):
    #compile patern into pattern objects 
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    #it will compare pattern object with tweet if pattern present in the tweet then remov i with space       
    return url_pattern.sub(r'', text)   

#it will create the pattern from smileys
def make_emoticon_pattern(emoticons):                
    pattern = "|".join(map(re.escape, emoticons.Smiley))
    pattern = "(?<=\s)(" + pattern + ")(?=\s)"
    return pattern

#it will replace the smile with ''
def find_with_pattern(pattern,data,replace=False, tag=None):   
    if replace and tag == None:                           
        raise Exception("Parameter error", "If replace=True you should add the tag by which the pattern will be replaced")
    regex = re.compile(pattern)
    if replace:
        #it will replace the pattern with tag
        return data.apply(lambda tweet: re.sub(pattern, tag, " " + tweet + " "))
    #it will find all the smiley in the tweets
    return data.apply(lambda tweet: re.findall(pattern, " " + tweet + " "))

#Remove Html entities
def strip_html_tags(text):
    #Beautifulsoup is the python package for parsing html entities
    soup = BeautifulSoup(text, "html.parser")
    #convert these html entities into text
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

#remove unicodes
def remove_unicode(string):
    try:
        #it will ignore all the ascii code from string
        string = string.encode('ascii', 'ignore').decode("utf-8")
    except UnicodeDecodeError:
        pass
    return string

#remove all punctuations from tweets
def remove_punctuations(string):
    string = re.sub(r'[^\w\s]', '', string)
    return string

#removing multiple space from text
def remove_spaces(string):
    string = re.sub(r"^\s+", "", string)
    return string

#this function will replace all acronyms with their translation
def acronym_to_translation(tweet, acronyms_counter,punctuation,acronym_dictionary):
    table = str.maketrans(punctuation," " * len(punctuation))
    tweet = str(tweet).translate(table)
    words = tweet.split()
    new_words = []
    for i, word in enumerate(words):
        if word in acronym_dictionary:
            acronyms_counter[word] += 1
            new_words.extend(acronym_dictionary[word].split())
        else:
            new_words.append(word)
    return new_words

#replace sequence of repeated characters with two characters
pattern = re.compile(r'(.)\1*')

#it will reduce the tweet word which have a long same sequence of word
def reduce_sequence_word(word):
    return ''.join([match.group()[:2] if len(match.group()) > 2 else match.group() for match in pattern.finditer(word)])

#it will return the reduced tweet 
def reduce_sequence_tweet(tweet):
    return [reduce_sequence_word(word) for word in tweet]

 #removing stop words
def remove_stops(row):
    my_list = row
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

#doing stemming of words
def stem_list(row):
    my_list = row
    stem_list = [stemming.stem(word) for word in my_list]
    return (stem_list)

#remove remaining tokens that are not alphabetic
def remove_nonalpha(row):
    my_list = row
    remove_alpha = [word for word in my_list if word.isalpha()]
    return (remove_alpha)

#rejoin all cleaned tokens
def rejoin(row):
    mylist = row
    filtered_sentence = (" ").join(mylist)
    return filtered_sentence

#this will preprocess test tweet
def preprocessing_test_tweet(Input,acronyms_counter,punctuation,acronym_dictionary):
    Input = Input.lower()
    print(Input)
    Input = remove_urls(Input)
    print(Input)
    Input = strip_html_tags(Input)
    print(Input)
    Input = remove_unicode(Input)
    print(Input)
    Input = re.sub(r'@\w{1,}', '', Input)
    print(Input)
    Input = remove_punctuations(Input)
    print(Input)
    Input = remove_spaces(Input)
    print(Input)
    Input = acronym_to_translation(Input, acronyms_counter,punctuation,acronym_dictionary);
    print(Input)
    Input = reduce_sequence_tweet(Input)
    print(Input)
    Input = remove_stops(Input)
    print(Input)
    Input = stem_list(Input)
    print(Input)
    Input = remove_nonalpha(Input)
    print(Input)
    Input = rejoin(Input)
    print(Input)
    return Input

def browsefunc():

    root = tk1.ThemedTk()
    root.geometry('500x250')
    root.title("Result | Twitter Sentiment Analysis")
    root.iconbitmap('tweet-logo-real-ico.ico')
    root.get_themes()
    root.set_theme('equilux')
    root.resizable(0, 0)
    answer = tk.Text(root, width=55, height=15)

    answer.pack()

    root.configure(background="SeaGreen2")
    filename = filedialog.askopenfilename()
    pathlabel = tk.Label(text=filename)
    pathlabel.pack()
    print(filename)
    print(type(filename))
    with open(filename, mode='r') as f2:
        Input = f2.read()

    print(Input)
    
    #import emotions.csv file
    data = pd.read_csv('Dataset/emotions.csv', error_bad_lines=False)

    #convert uppercase into lowercase
    data.SentimentText = data.SentimentText.str.lower()

    #remove urls                     
    data.SentimentText = data.SentimentText.apply(lambda text: remove_urls(text))

    #import smileys.csv file 
    emoticons = pd.read_csv('Dataset/smileys.csv')
    positive_emoticons = emoticons[emoticons.Sentiment == 1]
    negative_emoticons = emoticons[emoticons.Sentiment == 0]

    pos_emoticons_found = find_with_pattern(make_emoticon_pattern(positive_emoticons),data.SentimentText)
    neg_emoticons_found = find_with_pattern(make_emoticon_pattern(negative_emoticons), data.SentimentText)

    #replace all emoticons with their ''
    data.SentimentText = find_with_pattern(make_emoticon_pattern(positive_emoticons),data.SentimentText, True, '')
    data.SentimentText = find_with_pattern(make_emoticon_pattern(negative_emoticons),data.SentimentText, True, '')

    #Remove Html entities
    data.SentimentText = data.SentimentText.apply(lambda text: strip_html_tags(text))

    #remove unicodes
    data.SentimentText = data.SentimentText.apply(lambda tweet: remove_unicode(tweet))

    #replace all @ with ''
    pattern_usernames = "@\w{1,}";
    data.SentimentText = find_with_pattern(pattern_usernames,data.SentimentText, True, '')

    #remove all punctuations from tweets
    data.SentimentText = data.SentimentText.apply(lambda tweet: remove_punctuations(tweet))

    #removing multiple space from text
    data.SentimentText = data.SentimentText.apply(lambda tweet: remove_spaces(tweet))

    #load set of acronyms
    acronyms = pd.read_csv('Dataset/acronyms.csv')

    #Create a dictionary of acronym which will be used to get translations
    acronym_dictionary = dict(zip(acronyms.Acronym, acronyms.Translation))

    #punctuations pattern
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{}~'

    #Frequency table for acronyms
    acronyms_counter = Counter()

    #this function will replace all acronyms with their translation
    data.SentimentText = data.SentimentText.apply(lambda tweet: acronym_to_translation(tweet, acronyms_counter,punctuation,acronym_dictionary));

    #replace sequence of repeated characters with two characters
    data.SentimentText = data.SentimentText.apply(lambda tweet: reduce_sequence_tweet(tweet))

    #removing stop words
    data.SentimentText = data.SentimentText.apply(lambda tweet: remove_stops(tweet))

    #doing stemming of words
    data.SentimentText = data.SentimentText.apply(lambda tweet: stem_list(tweet))

    #remove remaining tokens that are not alphabetic
    data.SentimentText = data.SentimentText.apply(lambda tweet: remove_nonalpha(tweet))

    #rejoin all cleaned tokens
    data.SentimentText = data.SentimentText.apply(lambda tweet: rejoin(tweet))

    #Before training, and even vectorizing, let's split our data into training and testing sets. 
    #It's important to do this before doing anything with the data so we have a fresh test set.
    X = data.SentimentText
    y = data.Sentiment

    #our test size is 0.2 which means 20% of test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #X_train_vect is now transformed into the right format to give to the Naive Bayes model
    vect = CountVectorizer(max_features=2000, binary=True)
    X_train_vect = vect.fit_transform(X_train)

    #One approach to addressing imbalanced datasets is to oversample the minority class.
    #Note: We have to make sure we only oversample the train data so we don't leak any information to the test set

    #this method will artificially grows minority classes and will balance every class 
    sm = SMOTE()
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

    #The classes are now balanced for the train set. We can move onto training a Naive Bayes model.
    unique, counts = np.unique(y_train_res, return_counts=True)

    #now train the model using multinomial naive bayes 
    nb = MultinomialNB()

    #fit the model
    nb.fit(X_train_res, y_train_res)

    #we need to use our test set in order to get a good estimate of accuracy.
    #Let's vectorize the test set

    #y_pred now contains a prediction for every row of the test set. With this prediction result, we can pass it into an 
    #sklearn metric with the true labels to get an accuracy score, F1 score, and generate a confusion matrix:
    X_test_vect = vect.transform(X_test)
    y_pred = nb.predict(X_test_vect)

    #for improving accuracy we will do Kfold's validation
    X = data.SentimentText
    y = data.Sentiment

    #This performs a shuffle first and then a split of the data into train/test. Since it's an iterator, it will perform a random 
    #shuffle and split for each iteration.
    ss = ShuffleSplit(n_splits=10, test_size=0.2)
    sm = SMOTE()

    accs = []
    f1s = []
    cms = []

    for train_index, test_index in ss.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit vectorizer and transform X train, then transform X test
        X_train_vect = vect.fit_transform(X_train)
        X_test_vect = vect.transform(X_test)
        
        # Oversample
        X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
        
        # Fit Naive Bayes on the vectorized X with y train labels, 
        # then predict new y labels using X test
        nb.fit(X_train_res, y_train_res)
        y_pred = nb.predict(X_test_vect)
        
        # Determine test set accuracy and f1 score on this fold using the true y labels and predicted y labels
        accs.append(accuracy_score(y_test, y_pred))
        cms.append(confusion_matrix(y_test, y_pred))

    Input = preprocessing_test_tweet(Input,acronyms_counter,punctuation,acronym_dictionary);
    print(Input)
    ans = nb.predict(vect.transform([Input]));
        # -> when classification done write if else code block like below...

    print(ans)
    print(type(ans))

    if(ans==[0]):
        My_Ans = "Tweet belongs to Sad Class!"  # -> Change this My_Ans to result 
    elif(ans==[1]):
        My_Ans = "Tweet belongs to Worry Class!"  # -> Change this My_Ans to result 
    elif(ans==[2]):
        My_Ans = "Tweet belongs to Anger Class!"  # -> Change this My_Ans to result 
    elif(ans==[3]):
        My_Ans = "Tweet belongs to Happy Class!"  # -> Change this My_Ans to result

    answer.config(state='normal')
    answer.delete(1.0, tk.END)
    answer.insert(tk.INSERT, My_Ans)
    answer.config(state='disabled')

    root.mainloop()





root=tk1.ThemedTk()
root.geometry('1000x500')
root.title("Twitter Sentiment Analysis")
root.iconbitmap('tweet-logo-real-ico.ico')
root.get_themes()
root.set_theme('equilux')
root.resizable(0,0)

canvas=Canvas(root,width=1000,height=500)
image=ImageTk.PhotoImage(Image.open('bck.png'))  # -> Put Your Image Path Here..
img=canvas.create_image(0,0,anchor=NW,image=image)


w=ttk.Label(canvas,text="Twitter Sentiment Analysis",image=image,font='Helvetica 50 bold',foreground='black',compound='center')

button2 = ttk.Button(canvas, text="Browse-Tweets-Text-File",command=browsefunc)
button2.pack(in_=canvas,side=BOTTOM)
w.pack()
canvas.pack()


root.mainloop()