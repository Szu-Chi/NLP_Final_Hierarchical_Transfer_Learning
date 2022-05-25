import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
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

METRICS = [
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro', name='micro_f1')
]


def make_model(embedding_matrix, num_tokens, embedding_dim, metrics=METRICS):
    int_sequences_input = keras.Input(shape=(300,), dtype="int64")
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
        loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS
    )
    return model

# ### Define Transfer Model
def make_trans_model(parent, embedding_matrix, num_tokens, embedding_dim):
    parent_model = make_model(embedding_matrix, num_tokens, embedding_dim)
    parent_model.load_weights(parent)
    child_model = keras.models.Model(inputs=parent_model.input, outputs=parent_model.layers[-2].output)
    new_out = keras.layers.Dense(1, activation='sigmoid', name='new_dense')(child_model.layers[-1].output)
    child_model = keras.models.Model(inputs=parent_model.input, outputs=new_out)
    child_model.layers[1].trainable=False
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=1e-3),
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