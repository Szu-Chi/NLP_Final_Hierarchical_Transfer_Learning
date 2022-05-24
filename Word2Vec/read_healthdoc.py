import csv
import numpy as np

def loading(dataset_path):
  id = np.empty(0)
  content = {}
  with open(dataset_path+"healthdoc_label.csv", newline='', encoding='utf-8') as f:
    rows = csv.reader(f, delimiter=',')
    print("Read healthdoc")
    for row in rows:
      id=np.append(id, row[0])
  for file in id[1:]:
    with open(dataset_path+file, 'r', encoding="utf-8") as f:
        content[file]=f.read()
  return content