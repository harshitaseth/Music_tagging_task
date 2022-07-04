from asyncore import read
import pandas as pd
import numpy as np
import csv
import pickle 

mood_csv = "/Users/harshita/Documents/Utopia/musimotion_testset_annotations_cut.csv"
def read_tsv(fn):
        r = []
        with open(fn) as tsv:
            reader = csv.reader(tsv, delimiter='\t')
            for row in reader:
                r.append(row)
        return r[0:]


df_mood = read_tsv(mood_csv)
tags_list = df_mood[0][1:]
mood_out = {}
for list_ in df_mood[1:]:
    mood_out[list_[0]] = {}
    indices = np.where(np.array(list_[1:])!='0.0')
    tags = np.array(tags_list)[indices[0]]
    mood_out[list_[0]]["tags"] = tags
    mood_out[list_[0]]["value"] = np.array(list_[1:])[indices[0]]

pickle.dump(mood_out, open("data_mapping.pkl","wb"))
keys = [*mood_out.keys()]
train_ids = keys[:int(len(mood_out.keys()) * 0.80)]
val_ids = keys[int(len(mood_out.keys()) * 0.80):]
pickle.dump(train_ids, open("train_ids.pkl","wb"))
pickle.dump(val_ids, open("val_ids.pkl","wb"))
pickle.dump(tags_list, open("All_tags.pkl","wb"))

