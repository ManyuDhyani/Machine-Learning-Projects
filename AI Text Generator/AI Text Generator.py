#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import string
import requests


# In[2]:


response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')


# In[3]:


data = response.text.split('\n')


# In[4]:


data = data[253:]


# In[5]:


len(data)


# In[6]:


data = " ".join(data)
data[:1000]


# In[7]:


def clean_text(doc):
  tokens = doc.split()
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens

tokens = clean_text(data)
print(tokens[:50])


# In[8]:


len(set(tokens))


# In[9]:


length = 50 + 1
lines = []

for i in range(length, len(tokens)):
  seq = tokens[i-length:i]
  line = ' '.join(seq)
  lines.append(line)
  if i > 200000:
    break

print(len(lines))


# In[10]:


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[11]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# In[12]:


sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:,-1]
X[0]


# In[13]:


y[0]


# In[1]:


vocab_size = len(tokenizer.word_index) + 1


# In[15]:


y = to_categorical(y, num_classes=vocab_size)


# In[16]:


X.shape[1]


# In[17]:


seq_length = X.shape[1]
seq_length


# In[18]:


model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))


# In[19]:


model.summary()


# In[20]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[24]:


model.fit(X, y, batch_size = 256, epochs = 1)


# In[25]:


seed_text=lines[12343]
seed_text


# In[26]:


def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')

    y_predict = model.predict_classes(encoded)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)


# In[27]:


generate_text_seq(model, tokenizer, seq_length, seed_text, 100)


# In[28]:


seed_text


# In[ ]:




