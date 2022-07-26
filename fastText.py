# code refer to https://keras.io/zh/examples/imdb_fasttext/

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score

# fastText
def make_model(cat_num, num_tokens, embedding_dim, max_length=300):
    print('Build fastText model...')

    int_sequences_input = keras.Input(shape=(max_length,), dtype="int64")
    embedded_sequences = keras.layers.Embedding(
        input_dim=num_tokens,
        output_dim=embedding_dim,
        input_length=max_length
    )(int_sequences_input)

    # average the embeddings of all words in the document
    global_average_layer = keras.layers.GlobalAveragePooling1D()(embedded_sequences)
    out = keras.layers.Dense(cat_num, activation='sigmoid')(global_average_layer)
    model = keras.models.Model(int_sequences_input, out)

    #    
    METRICS = [
        tfa.metrics.F1Score(num_classes=cat_num, threshold=0.5, average='micro', name='micro_f1')
    ] 
    model.compile(
        loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS
    )
    return model

def model_fit(model, x, y, val_data=None, class_weight=None):
    print('Start training fastText...')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_micro_f1', 
        verbose=1,
        patience=30,
        mode='max',
        restore_best_weights=True)
    
    if val_data != None:
        history = model.fit(x, y, batch_size=64, epochs=100, callbacks=[early_stopping], 
                   validation_data=val_data, class_weight=class_weight)
    else:
        history = model.fit(x, y, batch_size=64, epochs=100, callbacks=[early_stopping], 
                   validation_split=0.15, class_weight=class_weight)            
    return history

def get_model_result(model, test_x):
    y_pred = model.predict(test_x)
    output_shape = model.get_layer(index=-1).output_shape[1]
    if output_shape == 1: # if model is binary classifier
      y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])    
    else: # if model is multi-label classifier
      y_pred[y_pred>=0.5] = 1
      y_pred[y_pred<0.5] = 0
    return y_pred    

def calc_score(y_test, y_pred):
    num_classes = y_test.shape[1]
    micro_f1_metrics = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='micro')
    macro_f1_metrics = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro')
    weighted_f1_metrics = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='weighted')
    
    micro_f1_metrics.update_state(y_test, y_pred)
    macro_f1_metrics.update_state(y_test, y_pred)
    weighted_f1_metrics.update_state(y_test, y_pred)

    micro_f1 = micro_f1_metrics.result()
    macro_f1 = macro_f1_metrics.result()
    weighted_f1 = weighted_f1_metrics.result()
    subset_acc = accuracy_score(y_test, y_pred, normalize=True)
    print(f'micro_f1   : {micro_f1: .4f}')
    print(f'macro_f1   : {macro_f1: .4f}')
    print(f'weighted_f1: {weighted_f1: .4f}')
    print(f'accuray    : {subset_acc: .4f}')
    return micro_f1, macro_f1, weighted_f1, subset_acc 


# Define some functions
def loadHealthdocPKL(healdoc_pkl_path):
  healthdoc_pkl = open(healdoc_pkl_path, "rb")
  total_size = pickle.load(healthdoc_pkl) # get size of healthdoc_pkl
  doc_ws_list = []
  for doc in range(total_size):
      doc_ws=pickle.load(healthdoc_pkl, encoding='utf-8')
      doc_ws_list.append(doc_ws)
  return(doc_ws_list)  
  
def remove_punctuation(healthdoc_ws):
  # healthdoc_ws : list with word segments. e.g. ['核災','食品','／','買到','「','核災區','」','食品',...] 
  
  # punctuation of en and zh
  punctuation_en = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
  punctuation_zh = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
  digits = "0123456789０１２３４５６７８９" 

  # remove punctuation
  temp = []
  healthdoc_rm_pun = []
  for doc in healthdoc_ws:
    temp = []
    for word in doc:
        if word in punctuation_en or word in punctuation_zh or word in digits:
            continue
        else:
            temp.append(word)
    healthdoc_rm_pun.append(temp)  
  return healthdoc_rm_pun

def build_word_index(healthdoc_ws, min_count=5):
  # healthdoc_ws : list with word segments. e.g. ['核災','食品','買到','核災區','食品',...]
  # min_count : Ignores all words with total frequency lower than this.

  # calculate the number of word in all documents
  word_count = {}
  for index, doc in enumerate(healthdoc_ws):
    for word in doc:
      if word not in word_count.keys():
        word_count[word] = 1
      else:
        word_count[word] = word_count[word] + 1

  # sort by the number of word
  word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))   

  # build word-index pair which index of word is based on the frequency of the word
  min_count = 5
  index = 1 # 0 for unknown words
  word_index = {}
  for word, count in word_count.items():
    if count < min_count:
      break
    word_index[word] = index  
    index = index + 1

  print('Original number of word = {}'.format(len(word_count)))
  print('After remove min_count number of word = {}'.format(len(word_index)))
  print('{} words are removed'.format(len(word_count) - len(word_index)))
  # print first n word-index pair
  print('first n word-index pair\n')
  for k, v in word_index.items():
    print('({}, {})'.format(k, v), end=' ')
    if v > 10:
      print('\n')
      break  
  return word_index   

def vectorlize(healthdoc_ws, word_index):
  vec_all = []
  for doc in healthdoc_ws:
    vec = []
    for word in doc:
      if word in word_index.keys():
        vec.append(word_index[word])
      else:
        vec.append(0) # index 0 for unknown words    
    vec_all.append(vec)    
  return vec_all  

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

def ngram_feature(x_train, x_test, num_tokens, max_len, ngram_range=1):
    # x_train : training data with shape (n_text, text), where n_text is the number of document in training data
    # x_test : testing data with shape (n_text, text), where n_text is the number of document in testing data
    # num_tokens : word count of the dataset
    # max_len : max length of input text
    # ngram_range : 1 > unigram (default)
    #         2 > bigram
    #         3 > trigram
    
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = num_tokens + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        num_tokens = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, x_test, num_tokens  

def get_vectorlized_data(x_train_keys, id_token):
    x_train = []
    for key in x_train_keys:
      x_train.append(id_token[key])
    return x_train    

             