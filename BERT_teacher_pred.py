
# ### Set up Library
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import health_doc
import gc
from imp import reload
from doc_preprocessing import get_data_from_kfold

# ### Import Model
import BERT
reload(BERT)
from BERT import model_load
from BERT import get_tokenizer, get_tokenized_data, calc_score

# ### Define Function 
def get_model_result(model, x):
    y_pred = model.predict(x={'input_ids': x['input_ids']})
    return np.array(y_pred[:,0])

def teacher_pred():
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

    # ### Cross validation predict
    teacher_pred={}
    for testing_time in range(K):
        subset_test = [testing_time]
        subset_train = np.delete(np.arange(K), subset_test)
        x_train, y_train = get_data_from_kfold(k_id, k_label, subset_train)
        x_test, y_test = get_data_from_kfold(k_id, k_label, subset_test)

        model_path = f'model/{subset_test[0]}/'

        x_test_vec = get_tokenized_data(x_test, dataset_content, tokenizer) 
        # ### Predict Result
        y_pred = np.zeros(y_test.shape)   
        for i, label_name in enumerate(dataset_label_name):
            print(label_name)
            tf.keras.backend.clear_session()
            print(model_path+label_name)
            model = model_load(model_path+label_name, 1)            
            y_pred[:, i] = get_model_result(model, x_test_vec)
            del model
            gc.collect()
        
        for x, y in zip(x_test, y_pred):
            teacher_pred[x] = y
        
        y_pred_th = y_pred
        y_pred_th[y_pred_th>=0.5] = 1
        y_pred_th[y_pred_th<0.5] = 0
        
        # ### Calculate Predict Reslut
        micro_f1, macro_f1, weighted_f1, subset_acc = calc_score(y_test, y_pred_th)

        label_f1=[]
        for i, label_name in enumerate(dataset_label_name):
            label_f1.append(f1_score(y_test[:,i], y_pred[:,i]))
            print(f'{label_name:<15}:{label_f1[-1]: .4f}')

    # ### Export teacher predict
    with open('id_teacher_predict','wb') as f:
        pickle.dump(teacher_pred, f)