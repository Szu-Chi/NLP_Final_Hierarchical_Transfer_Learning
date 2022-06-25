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
from doc_preprocessing import get_data_from_kfold

import BERT
reload(BERT)
from BERT import make_model, model_fit, model_save, model_load
from BERT import get_tokenizer, get_tokenized_data, get_model_result, calc_score

# model
#   0: Normal multi-label classification
#   1: Knowledge Distillation
mode = 0

if (mode):
    # ### Get Teacher model prediction
    with open('id_teacher_predict','rb') as f:
        id_teacher_predict = pickle.load(f)

if __name__ == '__main__':
    # ### Loading HealthDoc dataset
    dataset_path = "HealthDoc/"
    dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)

    # ### Loading K-fold list
    with open('k_id', 'rb') as f:
        k_id = pickle.load(f)
    with open('k_label', 'rb') as f:
        k_label = pickle.load(f)

    K = len(k_id)

    tokenizer = get_tokenizer() # get BERT tokenizer

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

            model_path = f'/content/model/{subset_test[0]}/'

            # ### Training  Model
            #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

            # get tokenized data with BERT input format 
            x_train_vec = get_tokenized_data(x_train, dataset_content, tokenizer) 
            x_test = get_tokenized_data(x_test, dataset_content, tokenizer) 
            #x_val = getTokenized(x_val, dataset_content, tokenizer) 

            tf.keras.backend.clear_session()
            model = make_model(9)

            if (mode):
                y_train_teacher = np.empty(x_train.shape+(9,))
                for i, x in enumerate(x_train):
                    y_train_teacher[i,:] = id_teacher_predict[x]  
                
                print('Training Multi-label model with KD')
                history = model_fit(model, x_train_vec, y_train_teacher)
            else:
                print('Training Multi-label model without KD')
                history = model_fit(model, x_train_vec, y_train)
            
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