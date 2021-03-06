import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
sys.path.insert(0, '/host/src')

class SixDOFNet(nn.Module):
	def __init__(self):
		super(SixDOFNet, self).__init__()
		self.resnet = models.resnet18(pretrained=True)
		modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
		modules.append(nn.Dropout(0.5))
		self.resnet = nn.Sequential(*modules)
		#self.linear = nn.Linear(64512, out_features=3)
		#self.linear = nn.Linear(512, out_features=3)
		self.linear = nn.Linear(512, out_features=1)
	def forward(self, x):
		features = self.resnet(x)
		features = features.reshape(features.size(0), -1)
		features = self.linear(features)
		return features

if __name__ == '__main__':
	#model = SixDOFNet(3, 640, 480).cuda()
	#x = torch.rand((1,3,480,640)).cuda()
	model = SixDOFNet().cuda()
	print(model)
	x = torch.rand((1,3,200,200)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
