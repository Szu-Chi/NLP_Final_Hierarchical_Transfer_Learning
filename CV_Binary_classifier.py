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
import gc
from imp import reload
import doc_preprocessing
from doc_preprocessing import vectorlize, get_data_from_kfold

# ### Import Model
import GRU_att
reload(GRU_att)
from GRU_att import make_model, model_fit, calc_score
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
    return np.array(y_pred[:,0])

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
        # ### Create id_vector
        id_vector = {}
        for id, token in id_token.items():
            id_vector[id]=vectorlize(token, word_index)    

        # ### Training Category Model
        # class_weight = [1]
        class_weight = [0.05, 0.1, 0.5, 1, 2, 3, 5]
        for i, label_name in enumerate(dataset_label_name):
            x_train_vec = np.array([id_vector[x] for x in x_train])
            y_train_binary = y_train[:,i]
            x_train_vec, x_val_vec, y_train_binary, y_val_binary = train_test_split(x_train_vec, y_train_binary, 
                                                                test_size=0.15, stratify=y_train_binary)

            print(label_name)
            print(f'num of train positive: {np.where(y_train_binary==1)[0].size}')
            print(f'num of val positive: {np.where(y_val_binary==1)[0].size}')


            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_micro_f1', 
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)
            model_list = []
            val_micro_f1 = []
            history_list = []
            for cw in class_weight:
                tf.keras.backend.clear_session()
                model = make_model(1, embedding_matrix, num_tokens, embedding_dim)
                
                history = model_fit(model, x_train_vec, y_train_binary, 
                                    val_data=(x_val_vec, y_val_binary), class_weight={0:1, 1:cw})
                val_micro_f1.append(max(history.history['val_micro_f1']))
                model_list.append(model)
                history_list.append(history)
            best_model = model_list[np.argmax(val_micro_f1)]
            best_model_history = history_list[np.argmax(val_micro_f1)]
            save_model_history(best_model_history, label_name)
            best_model.save(model_path+label_name+'.h5')
            del model_list
            del best_model
            gc.collect()

        #%% [markdown]
        # ### Predict Result
        y_pred = np.zeros(y_test.shape)   
        for i, label_name in enumerate(dataset_label_name):
            print(label_name)
            tf.keras.backend.clear_session()
            print(model_path+label_name+'.h5')
            model = keras.models.load_model(model_path+label_name+'.h5')
            x_test_vec = np.array([id_vector[x] for x in x_test])
            y_pred[:, i] = get_model_result(model, x_test_vec)
            del model
            gc.collect()

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
