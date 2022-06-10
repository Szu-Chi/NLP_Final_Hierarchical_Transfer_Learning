import numpy as np

# ### Load pre-trained word embeddings
path_to_glove_file = './word_vector/healthdoc-wiki.vector'
embeddings_index = {}
with open(path_to_glove_file,'r', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

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

def get_data_from_kfold(k_id, k_label, index):
    x = np.empty(0)
    y = np.empty((0,9))
    subset_id = [k_id[i] for i in index]
    subset_label = [k_label[i] for i in index]
    for id, label in zip(subset_id, subset_label):
        x = np.append(x, id)
        y = np.append(y, label, axis=0)
    return x, y

def get_id_token(id_list, token_list):
    id_token={}
    for id, token in zip(id_list, token_list):
        id_token[id]=token
    return id_token

def get_voc(id_list, id_token):
    voc = ['', '[UKN]']
    for k in id_list:
        voc += id_token[k]
    voc = list(set(voc))
    voc = sorted(voc)
    return voc

def get_embedding_matrix(voc, word_index):
    num_tokens = len(voc) + 2
    embedding_dim = 300
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
    return embedding_matrix
