import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
import health_doc
import gc
from imp import reload
import doc_preprocessing
from doc_preprocessing import get_data_from_kfold


import fastText
reload(fastText)
from fastText import make_model, model_fit, get_model_result, calc_score
from fastText import loadHealthdocPKL, remove_punctuation, build_word_index, vectorlize, ngram_feature, get_vectorlized_data


# model
#   0: Normal multi-label classification
#   1: Knowledge Distillation
mode = 0

if (mode):
    # ### Get Teacher model prediction
    with open('fastText_id_teacher_predict','rb') as f:
        id_teacher_predict = pickle.load(f)

if __name__ == '__main__':
    # ### Loading HealthDoc dataset

    token_list_path = "/content/healthdoc.pkl"
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

    dataset_path = "/content/HealthDoc/"
    dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)
    #print (dataset_id[0], dataset_label[0], dataset_label_name, dataset_content[dataset_id[0]])

    id_token = doc_preprocessing.get_id_token(dataset_content.keys(), healthdoc_vec)

    # ### Loading K-fold list
    with open('k_id', 'rb') as f:
        k_id = pickle.load(f)
    with open('k_label', 'rb') as f:
        k_label = pickle.load(f)

    K = len(k_id)

    for cv_times in range(1):
        cv_micro_f1 = []
        cv_macro_f1 = []
        cv_accuray = []
        cv_weighted_f1 = []
        cv_label_f1 = []

        for testing_time in range(K):
            # ### Split data for train and test
            subset_test = [testing_time]
            subset_train = np.delete(np.arange(K), subset_test)
            x_train_keys, y_train = get_data_from_kfold(k_id, k_label, subset_train)
            x_test_keys, y_test = get_data_from_kfold(k_id, k_label, subset_test)

            model_path = f'/content/model/{subset_test[0]}/'

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

            # ### Training  Model
            tf.keras.backend.clear_session()
            model = make_model(9, num_tokens=num_tokens, embedding_dim=embedding_dims, max_length=max_len)
            if (mode):
                y_train_teacher = np.empty(x_train.shape+(9,))
                for i, x in enumerate(x_train):
                    y_train_teacher[i,:] = id_teacher_predict[x]                  
                print('Training Multi-label model with KD')
                history = model_fit(model, x_train, y_train_teacher)
            else:
                print('Training Multi-label model without KD')
                history = model_fit(model, x_train, y_train)
            
            gc.collect()

            # ### Predict Result
            y_pred = get_model_result(model, x_test)

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