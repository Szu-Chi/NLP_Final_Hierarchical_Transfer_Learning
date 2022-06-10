import csv 
import pickle
import numpy as np

def loading(dataset_path):
    id = np.empty(0)
    label  = np.empty((0,9))
    content = {}
    with open(dataset_path+"healthdoc_label.csv", newline='') as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            id=np.append(id, row[0])
            label=np.append(label, np.expand_dims(row[1:], axis=0), axis=0)
    for file in id[1:]:
        with open(dataset_path+file, 'r', encoding="utf-8") as f:
            content[file]=f.read()
    return id[1:], np.int16(label[1:]), content, label[0]
    
def loadHealthdocPKL(healdoc_pkl_path):
  healthdoc_pkl = open(healdoc_pkl_path, "rb")
  total_size = pickle.load(healthdoc_pkl) # get size of healthdoc_pkl
  doc_ws_list = []
  for doc in range(total_size):
      doc_ws=pickle.load(healthdoc_pkl, encoding='utf-8')
      doc_ws_list.append(doc_ws)
  return(doc_ws_list)
  