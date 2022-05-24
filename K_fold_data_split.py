#%% [markdown] 
# ### Set up Library
import pickle
import numpy as np
import matplotlib.pyplot as plt
import health_doc
import matplotlib.pyplot as plt

#%% [markdown]
# ### Loading HealthDoc dataset
dataset_path = "../dataset/HealthDoc/"
dataset_id, dataset_label, dataset_content, dataset_label_name = health_doc.loading(dataset_path)
print (dataset_id[0], dataset_label[0], dataset_label_name, dataset_content[dataset_id[0]])

#%% [markdown]
# ### Calculate Number of Label
def calc_label_num(label, label_name):
    label_num = {}
    for i, name in enumerate(label_name):
        label_num[name] = np.where(label[:,i]==1)[0].size
    label_num = {k: v for k, v in sorted(label_num.items(), key=lambda item: -item[1])}
    return label_num

label_num = calc_label_num(dataset_label, dataset_label_name)
for k, v in label_num.items():
    print(f'{k:<15} {v}')

plt.figure()
plt.bar(label_num.keys(), label_num.values())
plt.xticks(rotation=30, ha='right')
plt.title('dataset label distribution')

#%% [markdown]
# ### Split Data for K-fold Cross Validation
K = 5
def split_data(id, label, label_num):
    sorted_label = list(label_num.keys())
    sorted_label = sorted_label[::-1]
    print(sorted_label)
    temp_label = label
    temp_id = id
    k_id = [np.empty(0)]*K
    k_label = [np.empty((0,9))]*K
    for l in sorted_label:
        label_index = np.where(dataset_label_name==l)[0][0]
        id_index = np.where(temp_label[:, label_index]==1)[0]
        k_label_id = np.array_split(id_index,K)
        for i in range(K):
            k_id[i] = np.append(k_id[i], temp_id[k_label_id[i]])
            k_label[i] = np.append(k_label[i], temp_label[k_label_id[i],:], axis=0)
        temp_label = np.delete(temp_label, id_index, 0)
        temp_id = np.delete(temp_id, id_index, 0)
    return k_id, k_label
k_id, k_label = split_data(dataset_id, dataset_label, label_num)

for i, label in enumerate(k_label):
    label_num = calc_label_num(label, dataset_label_name)
    print('------------------------')
    for k, v in label_num.items():
        print(f'{k:<15} {v}')
    
    plt.figure()
    plt.bar(label_num.keys(), label_num.values())
    plt.xticks(rotation=30, ha='right')
    plt.title(f'subset{i+1} label distribution')

#%% [markdown]
# ### Check Data Split is Correct
for i in range(K):
    x = set(k_id[i])
    for j in range(K):
        if i == j:
            continue
        y = set(k_id[j])
        if x.intersection(y):
            raise ValueError('Subset intersection is not empty')

for subset_id, subset_label in zip(k_id, k_label):
    for id, label in zip(subset_id, subset_label):
        dataset_index = np.where(dataset_id==id)[0][0]
        if not np.array_equal(label, dataset_label[dataset_index]):
            raise ValueError('Subset Label and ID does not match')

#%% [markdown]
# ### Export k fold Data
with open('k_id', 'wb') as f:
    pickle.dump(k_id, f)
    
with open('k_label', 'wb') as f:
    pickle.dump(k_label, f)