import gc
import pickle
import numpy as np
from imp import reload
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

import health_doc
import doc_preprocessing
from doc_preprocessing import get_data_from_kfold

import fastText
reload(fastText)
from fastText import get_model_result, loadHealthdocPKL, remove_punctuation, build_word_index, vectorlize, ngram_feature, get_vectorlized_data


# ### Define Function 
def get_model_result(model, x):
    y_pred = model.predict(x)
    return np.array(y_pred[:,0])

def teacher_pred():
    # ### Loading HealthDoc dataset

    token_list_path = "healthdoc.pkl"
    healthdoc_ws = loadHealthdocPKL(token_list_path)
    healthdoc_ws = remove_punctuation(healthdoc_ws)

    # Set parameters:
    # ngram_range = 2 will add bi-grams features
    ngram_range = 1
    embedding_dims = 50
    min_count = 3
    max_len = 300
        
    word_index = build_word_index(healthdoc_ws, min_count)
    healthdoc_vec = vectorlize(healthdoc_ws, word_index)

    num_tokens = len(word_index)

    dataset_path = "../dataset/HealthDoc/"
    dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)

    id_token = doc_preprocessing.get_id_token(dataset_content.keys(), healthdoc_vec)
    
    # ### Loading K-fold list
    with open('k_id', 'rb') as f:
        k_id = pickle.load(f)
    with open('k_label', 'rb') as f:
        k_label = pickle.load(f)

    K = len(k_id)

    # ### Cross validation predict
    teacher_pred={}
    for testing_time in range(K):
        # ### Split data for train and test
        subset_test = [testing_time]
        subset_train = np.delete(np.arange(K), subset_test)
        x_train_keys, y_train = get_data_from_kfold(k_id, k_label, subset_train)
        x_test_keys, y_test = get_data_from_kfold(k_id, k_label, subset_test)

        model_path = f'model/{subset_test[0]}/'

        # Prepare train/test data
        x_train_vec = get_vectorlized_data(x_train_keys, id_token)
        x_test_vec = get_vectorlized_data(x_test_keys, id_token) 

        print(len(x_train_vec), 'train sequences')
        print(len(x_test_vec), 'test sequences')
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train_vec)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test_vec)), dtype=int)))
        x_train, x_test, num_tokens = ngram_feature(x_train_vec, x_test_vec, num_tokens, max_len, ngram_range=ngram_range)

        # ### Predict Result
        y_pred = np.zeros(y_test.shape)   
        for i, label_name in enumerate(dataset_label_name):
            print(label_name)
            tf.keras.backend.clear_session()
            print(model_path+label_name)
            model = keras.models.load_model(model_path+label_name+'.h5')                        
            y_pred[:, i] = get_model_result(model, x_test)
            del model
            gc.collect()
        
        for x, y in zip(x_test_keys, y_pred):
            teacher_pred[x] = y
        
        y_pred_th = y_pred
        y_pred_th[y_pred_th>=0.5] = 1
        y_pred_th[y_pred_th<0.5] = 0
        
        label_f1=[]
        for i, label_name in enumerate(dataset_label_name):
            label_f1.append(f1_score(y_test[:,i], y_pred[:,i]))
            print(f'{label_name:<15}:{label_f1[-1]: .4f}')

    # ### Export teacher predict
    with open('fastText_id_teacher_predict','wb') as f:
        pickle.dump(teacher_pred, f)