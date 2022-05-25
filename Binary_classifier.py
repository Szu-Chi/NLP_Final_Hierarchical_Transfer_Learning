#%% [markdown] 
# ### Set up Library
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import health_doc
import matplotlib.pyplot as plt
import jieba
from ckiptagger import WS
ws = WS("./ckiptagger_data")
import gc
from imp import reload

#%% [markdown]
# ### Loading HealthDoc dataset
dataset_path = "../dataset/HealthDoc/"
dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)
print (dataset_id[0], dataset_label[0], dataset_label_name, dataset_content[dataset_id[0]])

#%% [markdown]
# ### Loading K-fold list
with open('k_id', 'rb') as f:
    k_id = pickle.load(f)
with open('k_label', 'rb') as f:
    k_label = pickle.load(f)
K = len(k_id)
#%% [markdown]
# ### Select Train data and Test data
def get_subset_data(k_id, k_label, index):
    x = np.empty(0)
    y = np.empty((0,9))
    subset_id = [k_id[i] for i in index]
    subset_label = [k_label[i] for i in index]
    for id, label in zip(subset_id, subset_label):
        x = np.append(x, id)
        y = np.append(y, label, axis=0)
    return x, y

subset_test = [0]
subset_train = np.delete(np.arange(K), subset_test)
train_x, train_y = get_subset_data(k_id, k_label, subset_train)
test_x, test_y = get_subset_data(k_id, k_label, subset_test)

#%% [markdown]
# ### Get Train Content
train_content = {}
for x in train_x:
    train_content[x] = dataset_content[x]

#%% [markdown]
# ### Get token
voc = ['', '[UKN]']
id_token={}
for k, v in dataset_content.items():
    # id_token[k] = ws([v])[0]
    id_token[k] = list(jieba.cut(v, cut_all=False, HMM=False))
    if k in train_x:
        voc += id_token[k]
    pass
voc = list(set(voc))
print(voc[0:100])
word_index = dict(zip(voc, range(len(voc))))

# %% [markdown]
# ### Load pre-trained word embeddings
path_to_glove_file = './word_vector/healthdoc-wiki.vector'
embeddings_index = {}
with open(path_to_glove_file,'r', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

#%% [markdown]
# ### Prepare a corresponding embedding matrix
num_tokens = len(voc) + 2
embedding_dim = 300
hits = 0
misses = 0

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# %% [markdown]
# ### Loading pre-trained word embeddings matrix into an Embedding Layer
from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

# %% [markdown]
# ### Define HTrans Model
import model
reload(model)
from model import make_model

#%% [markdown]
# ### Define Vectorlize
def vectorlize(token, word_index):
    max_len = 300
    vec = []
    for t in token:
        if t in word_index.keys():
            vec.append(word_index[t])
        else:
            vec.append(word_index['[UKN]'])
    vec.append(word_index[''])
    if len(vec) < max_len:
        pad = np.zeros(max_len)
        pad[0:len(vec)]=vec
        return pad
    return np.array(vec[0:max_len])
    
id_vector = {}
for id, token in id_token.items():
    id_vector[id]=vectorlize(token, word_index)    


#%% [markdown]
# ### Define funtion to save model history
def save_model_history(history, topics):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(topics +' Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig('img/loss/subset/'+topics+'_loss.png')

    plt.figure()
    plt.plot(history.history['micro_f1'])
    plt.plot(history.history['val_micro_f1'])
    plt.title(topics +' Model micro_f1')
    plt.ylabel('micro_f1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig('img/micro_f1/subset/'+topics+'_micro_f1.png')


# %% [markdown]
# ### Training Category Model
class_weight = [1]
# class_weight = [1, 2, 3, 5, 10, 30, 50]
for i, label_name in enumerate(dataset_label_name):
    x = np.array([id_vector[x] for x in train_x])
    y = train_y[:,i]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, stratify=y)

    print(label_name)
    print(f'num of train positive: {np.where(y_train==1)[0].size}')
    print(f'num of val positive: {np.where(y_val==1)[0].size}')


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_micro_f1', 
        verbose=1,
        patience=3,
        mode='max',
        restore_best_weights=True)
    model_list = []
    val_micro_f1 = []
    history_list = []
    for cw in class_weight:
        tf.keras.backend.clear_session()
        model = make_model(embedding_matrix, num_tokens, embedding_dim)
        # img_path='network_image.png'
        # keras.utils.plot_model(model, to_file=img_path)
        # model.summary()
        history = model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stopping],
                            validation_split=0.15, class_weight = {0: 1, 1:cw})
        val_micro_f1.append(max(history.history['val_micro_f1']))
        model_list.append(model)
        history_list.append(history)
    best_model = model_list[np.argmax(val_micro_f1)]
    best_model_history = history_list[np.argmax(val_micro_f1)]
    save_model_history(best_model_history, label_name)
    best_model.save_weights('model/'+label_name+'.h5')
    del model_list
    del best_model
    gc.collect()

#%% [markdown]
# ### Define funtion to get predict result
def get_model_result(model, test_x):
    x = np.array([id_vector[x] for x in test_x])
    y_pred = model.predict(x)
    y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
    return y_pred

#%% [markdown]
# ### Predict Result
pred_y = np.zeros(test_y.shape)   
for i, label_name in enumerate(dataset_label_name):
    print(label_name)
    tf.keras.backend.clear_session()
    model = make_model(embedding_matrix, num_tokens, embedding_dim)
    model.load_weights(f'model/{label_name}.h5')
    pred_y[:, i] = get_model_result(model, test_x)
    del model
    gc.collect()

#%% [markdown]
# ### Calculate Predict Reslut
num_classes = test_y.shape[1]
micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
weighted_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='weighted')
accuray = keras.metrics.Accuracy()
micro_f1.update_state(test_y, pred_y)
macro_f1.update_state(test_y, pred_y)
weighted_f1.update_state(test_y, pred_y)

print(f'micro_f1   : {micro_f1.result(): .4f}')
print(f'macro_f1   : {macro_f1.result(): .4f}')
print(f'weighted_f1: {weighted_f1.result(): .4f}')

label_f1=[]
for i, label_name in enumerate(dataset_label_name):
    label_f1.append(f1_score(test_y[:,i], pred_y[:,i]))
    print(f'{label_name:<15}:{label_f1[-1]: .4f}')
plt.figure()
plt.bar(dataset_label_name, label_f1)
plt.xticks(rotation=30, ha='right')
plt.title(f'label micro f1')