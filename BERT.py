'''
code refer to : 
https://towardsdatascience.com/multi-label-multi-class-text-classification-with-bert-transformer-and-keras-c6355eccb63a

The BERT authors recommend some hyperparameter options for fine-tuning
  • Batch size: 16, 32
  • Learning rate (Adam): 5e-5, 3e-5, 2e-5
  • Number of epochs: 2, 3, 4
'''
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from transformers import BertTokenizer, BertModel
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.initializers import TruncatedNormal

max_length = 300
model_name = 'hfl/chinese-bert-wwm' # 哈工大BERT

def make_model(cat_num, max_length=max_length, model_name=model_name):
    METRICS = [
        tfa.metrics.F1Score(num_classes=cat_num, threshold=0.5, average='micro', name='micro_f1')
    ]    
    ### --------- Setup BERT ---------- ###
    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config) # Load BERT tokenizer
    transformer_model = TFBertModel.from_pretrained(model_name, config = config) # Load the Transformers BERT model
      
    ### ------- Build the model ------- ###
    bert = transformer_model.layers[0] # Load the MainLayer

    input_ids = keras.Input(shape=(100,), dtype='int64', name='input_ids')
    # attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    # inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    inputs = {'input_ids': input_ids}
    bert_model = bert(input_ids)[1]
    dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='pooled_output')
    pooled_output = dropout(bert_model, training=False)
    
    outputs = keras.layers.Dense(cat_num, kernel_initializer=TruncatedNormal(stddev=config.initializer_range), activation='sigmoid')(pooled_output)
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel')

    # pre-trained BERT (layer[1]) with less learning rate, 
    # classifier (layer[-1]) with larger learning rate
    optimizers = [
        keras.optimizers.Adam(learning_rate=3e-05),
        keras.optimizers.Adam(learning_rate=1e-04)
    ]
    optimizers_for_layers = [(optimizers[0], model.layers[1]), (optimizers[1], model.layers[-1])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_for_layers) 

    model.compile(
      loss="binary_crossentropy", 
      #optimizer=keras.optimizers.Adam(learning_rate=3e-05),
      optimizer=optimizer,
      metrics=METRICS
    )
    return model

def model_fit(model, x, y, val_data=None, class_weight=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_micro_f1', 
        verbose=1,
        patience=3,
        mode='max',
        restore_best_weights=True)
    if val_data != None:
        history = model.fit(
            {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}, y,
            batch_size=16, epochs=5, callbacks=[early_stopping], validation_data=val_data, class_weight=class_weight)
    else:
        history = model.fit({'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}, y,
            batch_size=16, epochs=5, callbacks=[early_stopping], validation_split=0.15, class_weight=class_weight)
    return history

def model_save(model, path):
    model.save_weights(path)
    print('save model to "{}"'.format(path))

def model_load(path, cat_num):
    model = make_model(cat_num)
    model.load_weights(path)  
    return model  

def get_model_result(model, x):
    y_pred = model.predict(x={'input_ids': x['input_ids']})

    output_shape = model.get_layer(index=-1).output_shape[1]
    if output_shape == 1: # if model is binary classifier
      y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])    
    else: # if model is multi-label classifier
      y_pred[y_pred>=0.5] = 1
      y_pred[y_pred<0.5] = 0
    return y_pred      


def getTokenized(dataset_id, dataset_content, tokenizer):
  # return tokenized data with BERT input format
  doc_content = []
  for name in dataset_id:
    doc_content.append(dataset_content[name])

  word_pieces = tokenizer(
      text=doc_content,  
      max_length=100,
      add_special_tokens=True,  # Add '[CLS]', '[SEP]', '[UNK]'  
      truncation=True,  # if token's lenght longer than 512 then truncate it
      padding=True, # if token's lenght less than 512 then padding it
      return_tensors='tf',
      return_token_type_ids = False,
      return_attention_mask = True)  
  return word_pieces    