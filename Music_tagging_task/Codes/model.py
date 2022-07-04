import torch
from torch import nn
from modules import Conv_2d, Conv_emb



class AudioModel(nn.Module):
	def __init__(self):
		super(AudioModel, self).__init__()

		# CNN module for spectrograms
		self.spec_bn = nn.BatchNorm2d(1)
		self.layer1 = Conv_2d(1, 128, pooling=2)
		self.layer2 = Conv_2d(128, 128, pooling=2)
		self.layer3 = Conv_2d(128, 256, pooling=2)
		self.layer4 = Conv_2d(256, 256, pooling=2)
		self.layer5 = Conv_2d(256, 256, pooling=2)
		self.layer6 = Conv_2d(256, 512, pooling=2)
		self.layer7 = Conv_2d(512, 512, pooling=2)
		self.layer8 = Conv_emb(512, 256)
		self.classifier = nn.Linear(256, 67)
		

	def spec_to_embedding(self, spec):
		
		
		out = spec.unsqueeze(1)
		out = self.spec_bn(out)

		# CNN
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		out = self.layer8(out)
		out = out.squeeze(2)
		out = nn.MaxPool1d(out.size(-1))(out)
		out = out.view(out.size(0), -1)
		classification = self.classifier(out)
	
		return classification

	

	def forward(self,spec):
		
		out = self.spec_to_embedding(spec)
		return out