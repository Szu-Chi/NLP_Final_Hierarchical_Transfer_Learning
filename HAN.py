# code reference :
#   1. https://github.com/ShawnyXiao/TextClassification-Keras#6-han
#   2. https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing import sequence
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from sklearn.metrics import accuracy_score

maxlen_word = 300
maxlen_sentence = 16

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

def make_model(cat_num, embedding_matrix, num_tokens, embedding_dim, maxlen_word=maxlen_word, maxlen_sentence=maxlen_sentence):
    METRICS = [
        tfa.metrics.F1Score(num_classes=cat_num, threshold=0.5, average='micro', name='micro_f1')
    ]
    # input layer
    int_sequences_input = keras.Input(shape=(maxlen_word,), dtype="int64") # maxlen_word = length of word vector
    # embedding layer
    embedded_sequences = keras.layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix.copy()),
        trainable=False,
    )(int_sequences_input)

    # Word encoder
    word_bidirectional = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(embedded_sequences)  # LSTM or GRU
    #word_bidirectional = keras.layers.Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedded_sequences)  # LSTM or GRU
    word_att = Attention(maxlen_word)(word_bidirectional) # maxlen_word = 300
    model_word_encoder = keras.models.Model(int_sequences_input, word_att)

    # Sentence encoder
    sent_input = keras.Input(shape=(maxlen_sentence, maxlen_word), dtype='int64')
    word_encoder = keras.layers.TimeDistributed(model_word_encoder)(sent_input)
    sentence_bidirectional = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(word_encoder)  # LSTM or GRU
    #sentence_bidirectional = keras.layers.Bidirectional(CuDNNLSTM(128, return_sequences=True))  # LSTM or GRU
    sentence_att = Attention(maxlen_sentence)(sentence_bidirectional)

    # output layer
    out = keras.layers.Dense(cat_num, activation='sigmoid')(sentence_att)

    model = keras.models.Model(sent_input, out)
    model.compile(
        loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS
    )
    return model

def model_fit(model, x, y, val_data=None, class_weight=None, maxlen_word=maxlen_word, maxlen_sentence=maxlen_sentence): 
    x = sequence.pad_sequences(x, maxlen=maxlen_sentence * maxlen_word)
    x = x.reshape((len(x), maxlen_sentence, maxlen_word))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_micro_f1', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    if val_data != None:
        val_data = list(val_data)
        val_data[0] = sequence.pad_sequences(val_data[0], maxlen=maxlen_sentence * maxlen_word)
        val_data[0] = val_data[0].reshape((len(val_data[0]), maxlen_sentence, maxlen_word))
        val_data = tuple(val_data)
        history = model.fit(x, y, batch_size=32, epochs=30, callbacks=[early_stopping],
                        validation_data=val_data, class_weight=class_weight)
    else:
        history = model.fit(x, y, batch_size=32, epochs=30, callbacks=[early_stopping],
                        validation_split=0.15, class_weight=class_weight)
    return history 


def get_model_result_HAN(model, x, maxlen_word = maxlen_word, maxlen_sentence = maxlen_sentence):
    x = sequence.pad_sequences(x, maxlen=maxlen_sentence * maxlen_word)
    x = x.reshape((len(x), maxlen_sentence, maxlen_word))

    y_pred = model.predict(x)
    output_shape = model.get_layer(index=-1).output_shape[1]
    if output_shape == 1: # if model is binary classifier
      y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])    
    else: # if model is multi-label classifier
      y_pred[y_pred>=0.5] = 1
      y_pred[y_pred<0.5] = 0
    return y_pred


# ---- save and load model ---- #
'''
model.save_weights("../model/my_model")
new_model = make_model(9, embedding_matrix, num_tokens, embedding_dim)
new_model.load_weights("../model/my_model") 
'''

#   Attention layer class
from tensorflow.keras import backend as BK
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = BK.reshape(BK.dot(BK.reshape(x, (-1, features_dim)), BK.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = BK.tanh(e)

        a = BK.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= BK.cast(mask, BK.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= BK.cast(BK.sum(a, axis=1, keepdims=True) + BK.epsilon(), BK.floatx())
        a = BK.expand_dims(a)

        c = BK.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim