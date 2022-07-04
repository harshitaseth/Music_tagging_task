"""
load wav file and return spectrogram and corresponding label(tags)
"""

import os
import sys
import pickle
import tqdm
import numpy as np
import pandas as pd
import random
import librosa
from torch.utils import data
from scipy.io import wavfile
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class MyDataset(data.Dataset):
	def __init__(self, data_path, split='TRAIN'):
		
		self.split = split
		self.data_path = data_path
		self.dataset_mapping = pickle.load(open(os.path.join(data_path,"data_mapping.pkl"), 'rb'))
		self.song_datapath = "../Data/wavs/"
		self.all_tags = pickle.load(open(os.path.join(data_path,"All_tags.pkl"), 'rb'))
		self.labels = pickle.load(open(os.path.join(data_path,"label_dict.pkl"), 'rb'))

		# load ids
		if split == 'TRAIN':
			self.train_ids = pickle.load(open(os.path.join(data_path, 'new_train_ids.pkl'),"rb"))
		elif split == 'VALID':
			self.eval_ids =pickle.load(open(os.path.join(data_path, 'new_val_ids.pkl'),"rb"))
		


	def load_spec(self, song_id):
		
		fn = self.song_datapath + song_id
		sr, y = wavfile.read(fn)
		y = y.mean(axis =1) # reducing channel
		y = y / np.linalg.norm(y) #normalizing sample value
		
		full_length = len(y)//sr
		int_ = random.randint(1,full_length -32)
		# random croping of audio input
		y = y[int_*sr: (int_+30)* sr]
		y_out = np.zeros((30*sr))
		y_out[:len(y)] = y
	

		# appending with zeros if y_out < 30 sec
		spec = librosa.feature.melspectrogram(y_out)
		spec_out = np.zeros((128, 2813))
		if spec.shape[1] <2813:
			spec_out[:,:spec.shape[1]] = spec
		else:
			spec_out[:,:] = spec[:,:2813]
		
		return spec_out

	def get_train_item(self, index):

		train_id = self.train_ids[index//len(self.train_ids)]
		data = self.dataset_mapping[train_id.split(".")[0] + ".mp3"]
		
		tags = data["tags"]
		tags_label = []
		for tag in tags:
			tags_label.append(self.labels[tag])
		label = np.zeros(len(self.all_tags))
		label[tags_label] = 1
		
		spec = self.load_spec(train_id)
		return label, spec, train_id

	def get_eval_item(self, index):
	
		eval_id = self.eval_ids[index]
		data = self.dataset_mapping[eval_id.split(".")[0] + ".mp3"]
		
		tags = data["tags"]
		tags_label = []
		for tag in tags:
			tags_label.append(self.labels[tag])
		spec = self.load_spec(eval_id)
		label = np.zeros(len(self.all_tags))
		label[tags_label] = 1
		return label, spec, eval_id



	def __getitem__(self, index):
		if self.split == 'TRAIN':
			
			tags_label, spec, train_id = self.get_train_item(index)
			return  tags_label, spec.astype('float32'), train_id
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			tags_label, spec, eval_id= self.get_eval_item(index)

			return tags_label, spec.astype('float32'),eval_id

	def __len__(self):
		if self.split == 'TRAIN':
			return 5
		elif (self.split == 'VALID') or (self.split == 'TEST'):
			return len(self.eval_ids)


