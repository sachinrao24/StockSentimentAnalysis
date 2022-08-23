#!/usr/bin/env python
# coding: utf-8

# # Stock Sentiment Analysis based on News Headlines


import pandas as pd
import pickle


df = pd.read_csv('D:\Sachin\Mini Projects\Stock Sentiment Analysis\Data\Data.csv', encoding='ISO-8859-1')


train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']



#Removing non-alphabetic characters
data = train.iloc[:,2:27]
data.replace('[^a-zA-Z]', ' ', regex=True, inplace=True)


#Renaming columns to numerical index
idx_list = [i for i in range(25)]
new_index = [str(i) for i in idx_list]
data.columns = new_index



for index in new_index:
    data[index] = data[index].str.lower()
data.head(3)



combined_headlines = []
for row in range(0, len(data.index)):
    combined_headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))




from sklearn.ensemble import RandomForestClassifier
#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(2,2))
train_data = count_vectorizer.fit_transform(combined_headlines)
pickle.dump(count_vectorizer, open('count_vectorizer.pkl', 'wb'))



rfc = RandomForestClassifier(n_estimators=200, criterion='entropy')
rfc.fit(train_data, train['Label'])


test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_data = count_vectorizer.transform(test_transform)
predictions = rfc.predict(test_data)


# Saving model to disk
pickle.dump(rfc, open('rfc.pkl', 'wb'))

    
    
    
