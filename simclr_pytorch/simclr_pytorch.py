import torch
from torch.nn import nn
from torchvision import datasets, transforms
from torchlars import LARS # pip install torchlars
from torch.optim import SGD

torch.manual_seed(17)

class ContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        
    def similarity(self, zi, zj):
        return torch.dot(zi, zj) / (torch.norm(zi) * torch.norm(zj))
        
    def forward(self, zi, zj):
        '''
        Normalised Temperature-scaled Cross Entropy Loss
        '''
        return -torch.log(torch.softmax(self.similarity(zi, zj) / self.temp))

class SimCLR(nn.Module):
    def __init__(self, image_size, indim, big_cnn):
        super().__init__()
        self.image_size = image_size
        self.big_cnn = big_cnn
        self.l1 = nn.Linear(indim, 512)
        self.l2 = nn.Linear(512, 128)
        
    def random_transform(self, x):
        T = transforms.RandomChoice([
            T.RandomCrop(),
            T.RandomGrayscale(),
            T.RandomResizedCrop(),
            T.RandomRotation(),
            T.RandomVerticalFlip(),
            T.GaussianBlur(),
            T.RandomInvert()
        ])
        
        return T(x)
        
    def forward(self, x):
        '''
        x is an image of size (image_size)
        '''
        # first augmentation
        x1 = self.random_transform(x)
        rep1 = self.big_cnn(x1)
        proj1 = self.l2(torch.relu(self.l1(rep1)))
        
        # second augmentation
        x2 = self.random_transform(x)
        rep2 = self.big_cnn(x2)
        proj2 = self.l2(torch.relu(self.l1(rep2)))
        
        return proj1, proj2
        
    def fit(self, train_loader):
        '''
        train_loader is the dataset loader with batch size B
        '''
        
        loss = ContrastiveLoss()
        base_optimizer = SGD(self.big_cnn.parameters(), lr=4.8)
        optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        
        for k, (data, label) in enumerate(train_loader, 0):
            proj1, proj2 = self.forward(data)
            
            optimizer.zero_grad()
            L = loss(proj1, proj2)
            L.backward()
            optimizer.step()
            
        return self.big_cnn # final self-supervised model