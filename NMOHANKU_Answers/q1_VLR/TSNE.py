"""
Created on Wed Feb 21 23:20:37 2024

@author: NIshanth Mohankumar
"""
import torch
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
from utils import *
from train_q2 import ResNet
from matplotlib import patches


import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def get_feats(model, test_loader):
    
    truncated_model = create_feature_extractor(model, return_nodes={'resnet': 'avgpool'})
    feats, targets = [], []
    for data, label, _ in test_loader:
        feat = truncated_model(data.to("cuda"))['avgpool'].reshape((data.shape[0], -1))
        # feats.append(feat.item().numpy())
        # targets.append(label.view(-1, 20).item().numpy())
        feats.append(feat.detach().cpu().numpy())
        targets.append(label.reshape(-1, 20).detach().cpu().numpy())
    return  np.concatenate(feats), np.concatenate(targets).astype(np.int32)

    
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "checkpoint-model-epoch10.pth"
    
    model = torch.load(model_path).to(device)
    model.eval()
    
    test_loader = get_data_loader('voc', train=False, batch_size=100, split='test')
    
    feats, targets = get_feats(model, test_loader)
    
    tsne = TSNE(n_components = 2)
    proj = tsne.fit_transform(feats) # Number_of_samples * 2
    
    colors = np.array([[np.random.choice(np.arange(256), size=3)] for i in range(20)])
    mean_colors = []
    for i in range(proj.shape[0]):
        colors1 = colors[np.where(targets[i, :]==1)]
        mean_colors.append(np.mean(colors1, axis=0, dtype=np.int32))

    plt.figure(figsize=(15,15))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.array(mean_colors)/255)
    plt.legend(handles=[patches.Patch(color=np.array(colors[i])/255, label="class " + str(i)) for i in range(20)])
    plt.title("TSNE")
    plt.savefig("feature_visualization.png")
    

if __name__ == '__main__':
    main()
   