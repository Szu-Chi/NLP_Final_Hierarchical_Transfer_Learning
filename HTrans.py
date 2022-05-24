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
path_to_glove_file = './word_vector/wiki.zh.vector'
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
embedding_dim = 400
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
METRICS = [
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro', name='micro_f1')
]

def make_model(metrics=METRICS):
    int_sequences_input = keras.Input(shape=(300,), dtype="int64")
    # embedded_sequences = embedding_layer(int_sequences_input)
    embedded_sequences = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix.copy()),
        trainable=False,
    )(int_sequences_input)

    forward_gru = keras.layers.GRU(96, return_sequences=True)
    backward_gru = keras.layers.GRU(96, return_sequences=True, go_backwards=True)
    bidirectional_layer = keras.layers.Bidirectional(forward_gru, backward_layer=backward_gru)(embedded_sequences)
    bidirectional_layer = keras.layers.Dropout(0.4)(bidirectional_layer)

    max_pooling_layer = keras.layers.MaxPooling1D(5)(bidirectional_layer)
    mean_pooling_layer = keras.layers.AveragePooling1D(5)(bidirectional_layer)
    attention_layer = keras.layers.Attention()([bidirectional_layer, bidirectional_layer])

    merged = keras.layers.concatenate([attention_layer, mean_pooling_layer, max_pooling_layer], axis=1)

    flatten_layer = keras.layers.Flatten()(merged)
    # linear_layer = keras.layers.Dense(200, activation='sigmoid')(flatten_layer)
    out = keras.layers.Dense(1, activation='sigmoid')(flatten_layer)

    model = keras.models.Model(int_sequences_input, out)
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=METRICS
    )
    return model

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

#%%  [markdown]
# ### Create Label Subset
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
# ### Training Subset Model
class_weight = [1]
# class_weight = [1, 2, 3, 5, 10, 30, 50]
for i, label_name in enumerate(dataset_label_name):
    x = np.array([id_vector[x] for x in train_x])
    y, subset_name = get_subset(train_y, label_name, label_subset)
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
        model = make_model()
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
    save_model_history(best_model_history, subset_name)
    best_model.save('model/'+subset_name+'.h5')
    del model_list
    del best_model
    gc.collect()

#%% [markdown]
# ### Define Transfer Model
def make_trans_model(parent):
    parent_model = keras.models.load_model('model/'+parent+'.h5')
    # parent_model.load_weights()
    child_model = keras.models.Model(inputs=parent_model.input, outputs=parent_model.layers[-2].output)
    new_out = keras.layers.Dense(1, activation='sigmoid', name='new_dense')(child_model.layers[-1].output)
    child_model = keras.models.Model(inputs=parent_model.input, outputs=new_out)
    child_model.layers[1].trainable=False
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=5e-4),
        tf.keras.optimizers.Adam(learning_rate=1e-3)
    ]

    optimizers_and_layers = [   (optimizers[0], child_model.layers[2]), 
                                (optimizers[0], child_model.layers[4]),
                                (optimizers[1], child_model.layers[-1])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    child_model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=METRICS
    )
    return child_model

# %% [markdown]
# ### Training Category Model
class_weight = [1]
# class_weight = [1, 2, 3, 5, 10, 30, 50]
for i, label_name in enumerate(dataset_label_name):
    x = np.array([id_vector[x] for x in train_x])
    subset_name = get_subset(train_y, label_name, label_subset)[1]
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
        model = make_trans_model(subset_name)
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
    model = make_model()
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
# =================================================================
# ==================Below Need to Complete=========================
# =================================================================
#%% [markdown]
# ### Define Transfer Model
def make_trans_model(parent):
    parent_model = make_model()
    parent_model.load_weights('model/'+parent+'.h5')
    child_model = keras.models.Model(inputs=parent_model.input, outputs=parent_model.layers[-2].output)
    new_out = keras.layers.Dense(1, activation='sigmoid', name='new_dense')(child_model.layers[-1].output)
    child_model = keras.models.Model(inputs=parent_model.input, outputs=new_out)
    child_model.layers[1].trainable=False
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=5e-4),
        tf.keras.optimizers.Adam(learning_rate=1e-3)
    ]

    optimizers_and_layers = [   (optimizers[0], child_model.layers[2]), 
                                (optimizers[0], child_model.layers[4]),
                                (optimizers[1], child_model.layers[-1])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    child_model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=METRICS
    )
    return child_model
#%% [markdown]
# ### Training Hierarchy 2 topics model
for key in hierarchy_topics[1]:
    parent = key
    child = hierarchy_topics[1][parent]
    for topics in child:
        x = vectorizer(np.array([[s] for s in train_data])).numpy()
        y = np.array([1 if topics in x else 0 for x in train_label])
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, stratify=y)
        print(topics)
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
            model = make_trans_model(parent)
            # img_path='network_image.png'
            # keras.utils.plot_model(model, to_file=img_path)
            # model.summary()
            history = model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stopping],
                                validation_data=(x_val, y_val),  class_weight = {0: 1, 1:cw})
            val_micro_f1.append(max(history.history['val_micro_f1']))
            model_list.append(model)
            history_list.append(history)
        
        best_model = model_list[np.argmax(val_micro_f1)]
        best_model_history = history_list[np.argmax(val_micro_f1)]
        save_model_history(best_model_history, topics)
        best_model.save_weights('model/'+topics+'.h5')
        del model_list
        del best_model
        gc.collect()

#%% [markdown]
# ### Predict Result  
category_test2 = np.empty((len(test_data),0))
category_pred2 = np.empty((len(test_data),0))
for key in hierarchy_topics[1]:
    parent = key
    child = hierarchy_topics[1][parent]
    for topics in child:
        print(topics)
        model = make_trans_model(parent)
        model.load_weights('model/'+topics+'.h5')
        y_test, y_pred = get_model_result(model, topics)
        category_test2 = np.hstack([category_test2, np.transpose([y_test])])
        category_pred2 = np.hstack([category_pred2, np.transpose([y_pred])])
        del model
        gc.collect()
np.save('category_test2', category_test2)
np.save('category_pred2', category_pred2)

#%% [markdown]
# ### Calculate Hierarchy 2 topics reslut
num_classes = category_pred2.shape[1]
micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
accuray = keras.metrics.Accuracy()
micro_f1.update_state(category_test2, category_pred2)
macro_f1.update_state(category_test2, category_pred2)
accuray.update_state(category_test2, category_pred2)

print(f'micro_f1: {micro_f1.result()}')
print(f'macro_f1: {macro_f1.result()}')
print(f'accuray: {accuray.result()}')

#%% [markdown]
# ### Training Hierarchy 3 topics model
for key in hierarchy_topics[2]:
    parent = key
    child = hierarchy_topics[2][parent]
    for topics in child:
        x = vectorizer(np.array([[s] for s in train_data])).numpy()
        y = np.array([1 if topics in x else 0 for x in train_label])
        print(topics)
        if np.where(y==1)[0].size > 1:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, stratify=y)
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
            model = make_trans_model(parent)
            # img_path='network_image.png'
            # keras.utils.plot_model(model, to_file=img_path)
            # model.summary()
            if np.where(y==1)[0].size > 1:
                history = model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stopping],
                                    validation_data=(x_val, y_val),  class_weight = {0: 1, 1:cw})
                val_micro_f1.append(max(history.history['val_micro_f1']))
            else:
                history = model.fit(x, y, batch_size=128, epochs=10,  class_weight = {0: 1, 1:cw})
                val_micro_f1.append(max(history.history['micro_f1']))
            model_list.append(model)
            history_list.append(history)

        best_model = model_list[np.argmax(val_micro_f1)]
        best_model_history = history_list[np.argmax(val_micro_f1)]
        if np.where(y==1)[0].size > 1:
            save_model_history(best_model_history, topics)
        best_model.save_weights('model/'+topics+'.h5')
        del model_list
        del best_model
        gc.collect()

#%% [markdown]
# ### Predict Result  
category_test3 = np.empty((len(test_data),0))
category_pred3 = np.empty((len(test_data),0))
for key in hierarchy_topics[2]:
    parent = key
    child = hierarchy_topics[2][parent]
    for topics in child:
        print(topics)
        model = make_trans_model(parent)
        model.load_weights('model/'+topics+'.h5')
        y_test, y_pred = get_model_result(model, topics)
        category_test3 = np.hstack([category_test3, np.transpose([y_test])])
        category_pred3 = np.hstack([category_pred3, np.transpose([y_pred])])
        del model
        gc.collect()
np.save('category_test3', category_test3)
np.save('category_pred3', category_pred3)

#%% [markdown]
# ### Calculate Hierarchy 3 topics reslut
num_classes = category_pred3.shape[1]
micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
accuray = keras.metrics.Accuracy()
micro_f1.update_state(category_test3, category_pred3)
macro_f1.update_state(category_test3, category_pred3)
accuray.update_state(category_test3, category_pred3)

print(f'micro_f1: {micro_f1.result()}')
print(f'macro_f1: {macro_f1.result()}')
print(f'accuray: {accuray.result()}')

#%% [markdown]
# ### Training Hierarchy 4 topics model
for key in hierarchy_topics[3]:
    parent = key
    child = hierarchy_topics[3][parent]
    for topics in child:
        x = vectorizer(np.array([[s] for s in train_data])).numpy()
        y = np.array([1 if topics in x else 0 for x in train_label])
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, stratify=y)
        print(topics)
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
            model = make_trans_model(parent)
            # img_path='network_image.png'
            # keras.utils.plot_model(model, to_file=img_path)
            # model.summary()
            history = model.fit(x_train, y_train, batch_size=128, epochs=10, callbacks=[early_stopping],
                            validation_data=(x_val, y_val),  class_weight = {0: 1, 1:cw})
            val_micro_f1.append(max(history.history['val_micro_f1']))
            model_list.append(model)
            history_list.append(history)

        best_model = model_list[np.argmax(val_micro_f1)]
        best_model_history = history_list[np.argmax(val_micro_f1)]
        save_model_history(best_model_history, topics)
        best_model.save_weights('model/'+topics+'.h5')
        del model_list
        del best_model
        gc.collect()

#%% [markdown]
# ### Predict Result  
category_test4 = np.empty((len(test_data),0))
category_pred4 = np.empty((len(test_data),0))
for key in hierarchy_topics[3]:
    parent = key
    child = hierarchy_topics[3][parent]
    for topics in child:
        print(topics)
        model = make_trans_model(parent)
        model.load_weights('model/'+topics+'.h5')
        y_test, y_pred = get_model_result(model, topics)
        category_test4 = np.hstack([category_test4, np.transpose([y_test])])
        category_pred4 = np.hstack([category_pred4, np.transpose([y_pred])])
        del model
        gc.collect()
np.save('category_test4', category_test4)
np.save('category_pred4', category_pred4)

#%% [markdown]
# ### Calculate Hierarchy 4 topics reslut
num_classes = category_test4.shape[1]
micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
accuray = keras.metrics.Accuracy()
micro_f1.update_state(category_test4, category_pred4)
macro_f1.update_state(category_test4, category_pred4)
accuray.update_state(category_test4, category_pred4)

print(f'micro_f1: {micro_f1.result()}')
print(f'macro_f1: {macro_f1.result()}')
print(f'accuray: {accuray.result()}')

#%% [markdown]
# ### Calculate all Hierarchy reslut
category_pred = np.hstack([category_pred1, category_pred2, category_pred3, category_pred4])
category_test = np.hstack([category_test1, category_test2, category_test3, category_test4])
num_classes = category_pred.shape[1]
micro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
macro_f1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
accuray = keras.metrics.Accuracy()
micro_f1.update_state(category_test, category_pred)
macro_f1.update_state(category_test, category_pred)
accuray.update_state(category_test, category_pred)

print(f'micro_f1: {micro_f1.result()}')
print(f'macro_f1: {macro_f1.result()}')
print(f'accuray: {accuray.result()}')