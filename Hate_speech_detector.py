#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
import fasttext
import fasttext.util
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[3]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[4]:


data = pd.read_csv('HateSpeechDatasetBalanced.csv', delimiter=',', header=None, names=['Content', 'Label'],low_memory=False)


# In[5]:


data.head()


# In[6]:


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# In[7]:


data['Content'] = data['Content'].apply(preprocess_text)


# In[8]:


data.head()


# In[9]:


model_path = "crawl-300d-2M.vec" #there is not this model in our project folder because it is too big for webonline submisson part
fasttext_model = KeyedVectors.load_word2vec_format(model_path, binary=False)


# In[10]:


print(fasttext_model.most_similar("hate"))  


# In[11]:


X = data['Content']  
y = data['Label']    


# In[12]:


X.head()


# In[13]:


def get_sentence_embedding(sentence, model):

    words = sentence.split()
    word_embeddings = [model.get_vector(word) for word in words if word in model]
    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)


# In[14]:


data['text_embedding'] = data['Content'].apply(lambda x: get_sentence_embedding(x, fasttext_model))
vectorized_data = np.vstack(data['text_embedding'].values) 


# In[15]:


le = LabelEncoder()
data['encoded_label'] = le.fit_transform(data['Label'])
labels = data['encoded_label']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(vectorized_data, labels, test_size=0.2, random_state=42)


# In[17]:


clf = LogisticRegression(max_iter=1000, random_state=42)  
clf.fit(X_train, y_train)


# In[18]:


y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[20]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Tahmin yap
y_pred = clf.predict(X_test)

# Performans metriklerini yazdır
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-hate", "hate"])

# Görselleştir
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format='g')
plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.tight_layout()
plt.show()


# In[ ]:




