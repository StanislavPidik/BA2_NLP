import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

# file name
file_name = os.path.basename(__file__)
# name without extension
file_name_without_ext = os.path.splitext(file_name)[0]
# part after the last underscore
last_underscore_index = file_name_without_ext.rfind('_')
version_str = file_name_without_ext[last_underscore_index + 1:]


OUTPUT_PATH = 'output'
# time_str = dt.now().strftime("%H-%M-%S")
date_str = dt.now().strftime("%Y_%m_%d")

if not os.path.exists(f'{OUTPUT_PATH}/{date_str}'):
   os.makedirs(f'{OUTPUT_PATH}/{date_str}')

data = pd.read_csv('./data_imdb/data_imdb_no_num.csv')


X_train, X_test, y_train, y_test = train_test_split(data['review_preproc'], 
                                                    data['label'], 
                                                    train_size=0.70, 
                                                    random_state=2)


top_words = 10000
max_review_length = 256
embedding_vecor_length = 64

tokenizer_lstm = Tokenizer(num_words=top_words)
tokenizer_lstm.fit_on_texts(X_train)
list_tokenized_train = tokenizer_lstm.texts_to_sequences(X_train)
X_train_lstm = pad_sequences(list_tokenized_train, maxlen=max_review_length)



def lr_schedule(epoch):
    lr = 0.01
    if epoch > 3:
        lr = 0.001
    if epoch > 7:
        lr = 0.0001
    return lr


optimizer = Adam(learning_rate=0.001) # old lr


model_lstm = Sequential()
model_lstm.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model_lstm.add(LSTM(32, return_sequences = True))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(32, activation = 'relu'))
model_lstm.add(Dense(1, activation='sigmoid')) 
model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_lstm.summary()

with open(f'{OUTPUT_PATH}/{date_str}/{dt.now().strftime("%Y-%m-%d")}_{dt.now().strftime("%H-%M-%S")}_lstm_model_summary_{version_str}.log', 'w') as f:
    model_lstm.summary(print_fn=lambda x: f.write(x + '\n'))

history_lstm = model_lstm.fit(X_train_lstm, y_train, epochs=12, batch_size=32, callbacks=[LearningRateScheduler(lr_schedule)])

history = history_lstm.history

df_hist = pd.DataFrame.from_dict(history) 
df_hist.to_csv (f'{OUTPUT_PATH}/{date_str}/{dt.now().strftime("%Y-%m-%d")}_{dt.now().strftime("%H-%M-%S")}_lstm_loos_accu_lr_{version_str}.csv', index = False, header=True, float_format='%.8f')

fig, ax1 = plt.subplots()
ax1.plot(history['loss'], label='train loss')
ax1.plot(history['accuracy'], label='train accuracy')
ax1.set_ylabel('Loss / Accuracy')
ax1.legend(loc='center left')
ax2 = ax1.twinx()
ax2.plot(history['lr'], label='learning rate', color='red')
ax2.set_yscale('log')
ax2.set_ylim([7e-05, 0.015])
ax2.set_ylabel('Learning Rate')
ax2.tick_params(axis='y', colors='red')
ax2.legend(loc='center right')
ax1.set_xlabel('Epoch')

plt.figure(figsize=(8,5))
fig.savefig(f'{OUTPUT_PATH}/{date_str}/{dt.now().strftime("%Y-%m-%d")}_{dt.now().strftime("%H-%M-%S")}_lstm_loos_accu_lr_{version_str}.png')
# plt.show()

model_lstm.save(f'lstm_model/{dt.now().strftime("%Y-%m-%d")}_{dt.now().strftime("%H-%M-%S")}_model_lstm_{version_str}.h5')

