import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, accuracy_score

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


METRICS = [
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro', name='micro_f1')
]

def make_model(cat_num, embedding_matrix, num_tokens, embedding_dim, metrics=METRICS):
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
    out = keras.layers.Dense(cat_num, activation='sigmoid')(flatten_layer)

    model = keras.models.Model(int_sequences_input, out)
    model.compile(
        loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS
    )
    return model


def model_fit(model, x, y, val_data=None, class_weight=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_micro_f1', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    if val_data != None:
        history = model.fit(x, y, batch_size=128, epochs=100, callbacks=[early_stopping],
                        validation_data=val_data, class_weight=class_weight)
    else:
        history = model.fit(x, y, batch_size=128, epochs=100, callbacks=[early_stopping],
                        validation_split=0.15, class_weight=class_weight)
    return history