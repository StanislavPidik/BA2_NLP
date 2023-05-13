import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data_imdb/data_imdb.csv')

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], 
    data['label'], 
    train_size=0.75, 
    test_size=0.25,
    random_state=2)

eval_texts, test_texts, eval_labels, test_labels = train_test_split(
    test_texts, 
    test_labels, 
    train_size=0.8, # train => eval => 20%
    test_size=0.2,  # -> 5%
    random_state=2)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts.values), truncation=True, padding=True)
val_encodings = tokenizer(list(eval_texts.values), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts.values), truncation=True, padding=True)

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])
history = model.fit(train_dataset.shuffle(1000).batch(4), epochs=4, batch_size=4, shuffle=True, verbose=1)

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./distilbert_fine_tuned_6/loss.png')
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./distilbert_fine_tuned_6/accuracy.png')
    plt.show()

plot_history(history)

model.save_pretrained('./distilbert_fine_tuned_6')
