#%% [markdown] 
# ### Set up Library
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
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
# ### Define Function 
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
        
def get_model_result(model, test_x):
    x = np.array([id_vector[x] for x in test_x])
    y_pred = model.predict(x)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return y_pred

def get_subset_data(k_id, k_label, index):
    x = np.empty(0)
    y = np.empty((0,9))
    subset_id = [k_id[i] for i in index]
    subset_label = [k_label[i] for i in index]
    for id, label in zip(subset_id, subset_label):
        x = np.append(x, id)
        y = np.append(y, label, axis=0)
    return x, y


# ### Define HTrans Model
import model
reload(model)
from model import make_model


def save_model_history(history, topics):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(topics +' Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig('img/loss/'+topics+'_loss.png')

    plt.figure()
    plt.plot(history.history['micro_f1'])
    plt.plot(history.history['val_micro_f1'])
    plt.title(topics +' Model micro_f1')
    plt.ylabel('micro_f1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig('img/micro_f1/'+topics+'_micro_f1.png')

label_subset = {'CAUTION'      :['DISE', 'HEALTH_PROTEC', 'INGRID'],
                'DISE'         :['CAUTION', 'HEALTH_PROTEC', 'TREAT'],
                'HEALTH_PROTEC':['CAUTION', 'DISE', 'INGRID'],
                'INGRID'       :['CAUTION', 'HEALTH_PROTEC', 'DISE'],
                'TREAT'        :['DISE', 'CAUTION', 'HEALTH_PROTEC'],
                'DRUG'         :['DISE', 'CAUTION', 'TREAT'],
                'MT_HEALTH'    :['DISE', 'CAUTION', 'HEALTH_PROTEC'],
                'EXAM'         :['DISE', 'CAUTION', 'TREAT'],
                'ELDER'        :['DISE', 'CAUTION', 'HEALTH_PROTEC'],
                }

def get_subset(train_y, label_name, label_subset):
    subset_name = label_name
    subset = label_subset[label_name]
    label_index = np.where(dataset_label_name==label_name)[0]
    print(label_index, label_name)
    y = train_y[:,label_index]
    print(f'num of train positive: {np.where(y==1)[0].size}')
    for label in subset:
        subset_name += "_"+label
        label_index = np.where(dataset_label_name==label)[0]
        y = np.logical_or(y, train_y[:,label_index])
        print(label_index, dataset_label_name[label_index])
        print(f'num of train positive: {np.where(y==1)[0].size}')
    return y, subset_name
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
#%%
def loadHealthdocPKL(healdoc_pkl_path):
  healthdoc_pkl = open(healdoc_pkl_path, "rb")
  total_size = pickle.load(healthdoc_pkl) # get size of healthdoc_pkl
  doc_ws_list = []
  for doc in range(total_size):
      doc_ws=pickle.load(healthdoc_pkl, encoding='utf-8')
      doc_ws_list.append(doc_ws)
  return(doc_ws_list)
#%% [markdown]
# ### Cross validation
with open('multi-times cv result.csv', 'w') as f:
    f.write('micro_f1,macro_f1,weighted_f1,subset_accuray,')
    for label_name in dataset_label_name:
        f.write(f'{label_name},')
    f.write('\n')
#%%
for cv_times in range(10):
    cv_micro_f1 = []
    cv_macro_f1 = []
    cv_accuray = []
    cv_weighted_f1 = []
    cv_label_f1 = []
    for testing_time in range(K):
        subset_test = [testing_time]
        subset_train = np.delete(np.arange(K), subset_test)
        train_x, train_y = get_subset_data(k_id, k_label, subset_train)
        test_x, test_y = get_subset_data(k_id, k_label, subset_test)

        model_path = f'model/{subset_test[0]}/'

        #%% [markdown]
        # ### Get Train Content
        train_content = {}
        for x in train_x:
            train_content[x] = dataset_content[x]

        #%% [markdown]
        # ### Get token
        voc = ['', '[UKN]']
        id_token={}
        token_list = loadHealthdocPKL('healthdoc.pkl')
        for k, t in zip(dataset_content.keys(), token_list):
            id_token[k] = t
            if k in train_x:
                voc += id_token[k]
            pass
        voc = list(set(voc))
        voc = sorted(voc)
        print(voc[0:100])
        word_index = dict(zip(voc, range(len(voc))))

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

        id_vector = {}
        for id, token in id_token.items():
            id_vector[id]=vectorlize(token, word_index)    

        # %% [markdown]
        # ### Training Subset Model
        x = np.array([id_vector[x] for x in train_x])
        y = train_y
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15)


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_micro_f1', 
            verbose=1,
            patience=3,
            mode='max',
            restore_best_weights=True)
        tf.keras.backend.clear_session()
        model = make_model(embedding_matrix, num_tokens, embedding_dim)
        # img_path='network_image.png'
        # keras.utils.plot_model(model, to_file=img_path)
        # model.summary()
        history = model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stopping],
                            validation_split=0.15)
        save_model_history(history, "multi label model")
        model.save_weights(model_path+'multi label model.h5')
        gc.collect()

        #%% [markdown]
        # ### Predict Result
        pred_y = get_model_result(model, test_x)

        #%% [markdown]
        # ### Calculate Predict Reslut
        num_classes = test_y.shape[1]
        micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
        macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
        weighted_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='weighted')
        micro_f1.update_state(test_y, pred_y)
        macro_f1.update_state(test_y, pred_y)
        weighted_f1.update_state(test_y, pred_y)
        subset_acc = accuracy_score(test_y, pred_y, normalize=True)
        
        cv_micro_f1.append(micro_f1.result())
        cv_macro_f1.append(macro_f1.result())
        cv_weighted_f1.append(weighted_f1.result())
        cv_accuray.append(subset_acc)

        print(f'micro_f1   : {micro_f1.result(): .4f}')
        print(f'macro_f1   : {macro_f1.result(): .4f}')
        print(f'weighted_f1: {weighted_f1.result(): .4f}')
        print(f'accuray    : {subset_acc: .4f}')

        label_f1=[]
        for i, label_name in enumerate(dataset_label_name):
            label_f1.append(f1_score(test_y[:,i], pred_y[:,i]))
            print(f'{label_name:<15}:{label_f1[-1]: .4f}')
        plt.figure()
        plt.bar(dataset_label_name, label_f1)
        plt.xticks(rotation=30, ha='right')
        plt.title(f'label micro f1')
        cv_label_f1.append(label_f1)
    with open('multi-times cv result.csv', 'a') as f:
        f.write(f'{sum(cv_micro_f1)/K: .4f},')
        f.write(f'{sum(cv_macro_f1)/K: .4f},')
        f.write(f'{sum(cv_weighted_f1)/K: .4f},')
        f.write(f'{sum(cv_accuray)/K: .4f},')
        
        label_f1_mean = np.mean(cv_label_f1, axis=0)
        for f1_mean in label_f1_mean:
            f.write(f'{f1_mean: .4f},')
        f.write('\n')

#%% [markdown]
# ### CV Result
print(f'micro_f1      : {sum(cv_micro_f1)/K: .4f}')
print(f'macro_f1      : {sum(cv_macro_f1)/K: .4f}')
print(f'weighted_f1   : {sum(cv_weighted_f1)/K: .4f}\n')
print(f'subset_accuray: {sum(cv_accuray)/K: .4f}\n')

label_f1_mean = np.mean(cv_label_f1, axis=0)
for label_name, f1_mean in zip(dataset_label_name, label_f1_mean):
    print(f'{label_name:<15}:{f1_mean: .4f}')

#%% [markdown]
# ### Export Result to CSV
with open('CV_result.csv', 'w') as f:
    f.write(', micro_f1, macro_f1, weighted_f1\n')
    for micro, marco, weighted in zip(cv_micro_f1, cv_macro_f1, cv_weighted_f1):
        f.write(f',{micro}, {marco}, {weighted} \n')
    f.write(f'Avg., {sum(cv_micro_f1)/K: .5f}, {sum(cv_macro_f1)/K: .5f}, {sum(cv_weighted_f1)/K: .5f}\n')

    for label_name in dataset_label_name:
        f.write(f',{label_name} ')
    f.write('\n')
    for label_f1 in cv_label_f1:
        for f1 in label_f1:
            f.write(f', {f1}')
        f.write('\n')
    label_f1_mean = np.mean(cv_label_f1, axis=0)
    f.write('Avg.,')
    for f1 in label_f1_mean:
        f.write(f'{f1: .5f}, ')
    f.write('\n')
