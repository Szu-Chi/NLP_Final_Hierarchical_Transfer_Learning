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
import gc
from imp import reload
import doc_preprocessing
from doc_preprocessing import vectorlize, get_data_from_kfold

# ### Import Model
import HAN
reload(HAN)
from HAN import make_model, model_fit, calc_score, get_model_result_HAN
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
def get_model_result(model, x):
    y_pred = model.predict(x)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return y_pred

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

#%% [markdown]
# ### Cross validation
with open('multi-times cv result.csv', 'w') as f:
    f.write('micro_f1,macro_f1,weighted_f1,subset_accuray,')
    for label_name in dataset_label_name:
        f.write(f'{label_name},')
    f.write('\n')
#%%
token_list = health_doc.loadHealthdocPKL('healthdoc.pkl')
for cv_times in range(10):
    cv_micro_f1 = []
    cv_macro_f1 = []
    cv_accuray = []
    cv_weighted_f1 = []
    cv_label_f1 = []
    for testing_time in range(K):
        # ### Split data for train and test
        subset_test = [testing_time]
        subset_train = np.delete(np.arange(K), subset_test)
        x_train, y_train = get_data_from_kfold(k_id, k_label, subset_train)
        x_test, y_test = get_data_from_kfold(k_id, k_label, subset_test)

        model_path = f'model/{subset_test[0]}/'

        #%% [markdown]
        # ### Prepare a corresponding embedding matrix
        id_token = doc_preprocessing.get_id_token(dataset_content.keys(), token_list)
        voc = doc_preprocessing.get_voc(x_train, id_token)
        print(voc[0:100])
        word_index = dict(zip(voc, range(len(voc))))

        embedding_matrix = doc_preprocessing.get_embedding_matrix(voc, word_index)
        num_tokens = len(voc) + 2
        embedding_dim = 300

        # %% [markdown]
        # ### Loading pre-trained word embeddings matrix into an Embedding Layer
        id_vector = {}
        for id, token in id_token.items():
            id_vector[id]=vectorlize(token, word_index)    

        # %% [markdown]
        # ### Training  Model
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

        tf.keras.backend.clear_session()
        model = make_model(9, embedding_matrix, num_tokens, embedding_dim)

        x_train_vec = np.array([id_vector[x] for x in x_train])
        x_val_vec = np.array([id_vector[x] for x in x_val])
        history = model_fit(model, x_train_vec, y_train, val_data=(x_val_vec, y_val))
        
        save_model_history(history, "multi label model")
        model.save_weights(model_path+'multi label model.h5')
        gc.collect()

        #%% [markdown]
        # ### Predict Result
        x_test_vec = np.array([id_vector[x] for x in x_test])
        y_pred = get_model_result_HAN(model, x_test_vec)

        #%% [markdown]
        # ### Calculate Predict Reslut
        micro_f1, macro_f1, weighted_f1, subset_acc = calc_score(y_test, y_pred)
        
        cv_micro_f1.append(micro_f1)
        cv_macro_f1.append(macro_f1)
        cv_weighted_f1.append(weighted_f1)
        cv_accuray.append(subset_acc)

        label_f1=[]
        for i, label_name in enumerate(dataset_label_name):
            label_f1.append(f1_score(y_test[:,i], y_pred[:,i]))
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
