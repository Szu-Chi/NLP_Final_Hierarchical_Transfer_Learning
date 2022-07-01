import pickle
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
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

from BERT_teacher_pred import teacher_pred   

if __name__ == '__main__':
    # ### Loading HealthDoc dataset
    dataset_path = "../dataset/HealthDoc/"
    dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)

    # ### Loading K-fold list
    with open('k_id', 'rb') as f:
        k_id = pickle.load(f)
    with open('k_label', 'rb') as f:
        k_label = pickle.load(f)

    K = len(k_id)
    tokenizer = get_tokenizer() # get BERT tokenizer

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
            x_train, y_train = get_data_from_kfold(k_id, k_label, subset_train)
            x_test, y_test = get_data_from_kfold(k_id, k_label, subset_test)

            model_path = f'model/{subset_test[0]}/'
            x_train_vec = get_tokenized_data(x_train, dataset_content, tokenizer) 
            x_test_vec = get_tokenized_data(x_test, dataset_content, tokenizer) 

            # ### Training Category Model
            # class_weight = [1]
            class_weight = [0.05, 0.1, 0.5, 1, 2, 3, 5]
            for i, label_name in enumerate(dataset_label_name):
                #x_train_vec, x_val_vec, y_train_binary, y_val_binary = train_test_split(x_train_vec, y_train_binary, test_size=0.15, stratify=y_train_binary)
                y_train_binary = y_train[:,i]

                print(label_name)

                val_micro_f1 = []
                # history_list = []
                for cw in class_weight:
                    tf.keras.backend.clear_session()
                    model = make_model(1)                
                    history = model_fit(model, x_train_vec, y_train_binary, class_weight={0:1, 1:cw})
                    val_micro_f1.append(max(history.history['val_micro_f1']))
                    # model_list.append(model)
                    # history_list.append(history)
                    model_save(model, f'model/temp/{label_name}_{cw}')
                    del model
                    gc.collect()
                best_model = np.argmax(val_micro_f1)
                model = model_load(f'model/temp/{label_name}_{class_weight[best_model]}', 1)
                # best_model_history = history_list[np.argmax(val_micro_f1)]
                #save_model_history(best_model_history, label_name)
                model_save(model, model_path+label_name)
                #best_model.save(model_path+label_name+'.h5')
                del model
                gc.collect()

            #%% [markdown]
            # ### Predict Result
            y_pred = np.zeros(y_test.shape)   
            for i, label_name in enumerate(dataset_label_name):
                print(label_name)
                tf.keras.backend.clear_session()
                #print(model_path+label_name+'.h5')
                model = model_load(model_path+label_name, 1)
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

    teacher_pred()