#%% [markdown]
# ### Setup Libirary
import tensorflow as tf
import health_doc
import pandas as pd
from datasets import Dataset
from transformers import TFAutoModelForCausalLM
from transformers import BertConfig, AutoTokenizer 
from transformers import create_optimizer, AdamWeightDecay

model_name = 'hfl/chinese-roberta-wwm-ext' # 哈工大BERT
#%% [markdown]
# ### Loading HealthDoc Dataset
dataset_path = "../dataset/HealthDoc/"
dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)

#%% [markdown]
# ### Convert HealthDoc to huggingface datasets format
df = pd.read_csv(f'{dataset_path}healthdoc_label.csv')
df['text'] = list(dataset_content.values())
health_doc_ds = Dataset.from_pandas(df)
health_doc_ds = health_doc_ds.train_test_split(test_size=0.2)
health_doc_ds = health_doc_ds.flatten()

#%% [markdown]
# ### Data Preprocessing
def get_training_corpus(dataset_id, dataset_content):
    i=0
    samples=[]
    for id in dataset_id:
        samples.append(dataset_content[id])
        i += 1
        if i == 100:
            yield samples
            i = 0
            samples=[]

def get_tokenizer(dataset_id, dataset_content, model_name=model_name):
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    # Load BERT tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, config = config) 
    training_corpus = get_training_corpus(dataset_id, dataset_content)
    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
    return tokenizer

tokenizer = get_tokenizer(dataset_id, dataset_content) # get BERT tokenizer

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]], truncation=True)

tokenized_health_doc_ds = health_doc_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=health_doc_ds["train"].column_names,
)


block_size = 128
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_health_doc_ds.map(group_texts, batched=True)


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")

tf_train_set = lm_dataset["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    dummy_labels=True,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = lm_dataset["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    dummy_labels=True,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

#%% [markdown]
# ### Define Model & Additional Pre-training
model = TFAutoModelForCausalLM.from_pretrained(model_name)
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)

#%% [markdown]
# ### Save pretrained model & tokenizer
model.save_pretrained('model/BERT_pretraining_with_HealthDoc/'+model_name)
tokenizer.save_pretrained('model/BERT_pretraining_with_HealthDoc/'+model_name)
