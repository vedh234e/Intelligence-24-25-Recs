import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformation=transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
path=r"C:\Users\Omarn\Documents\Code\Temp"
datasetimgs=datasets.ImageFolder(root=f'{path}/train',transform=transformation)
dataloader=DataLoader(datasetimgs,batch_size=32,shuffle=True,num_workers=4)
datasetlen=len(datasetimgs)
classess=datasetimgs.classes
model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
filters=model.classifier[1].in_features
model.classifier[1]=nn.Linear(filters, 1) 
model=model.to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)
def train_model(model,criterion,optimizer,num_epochs=15):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        lossiter=0.0
        correctruns=0
        for inputs, labels in dataloader:
            inputs, labels=inputs.to(device),labels.to(device)
            labels=labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            preds=torch.sigmoid(outputs).round()
            lossiter+=loss.item()*inputs.size(0)
            correctruns+=torch.sum(preds==labels.data)
        lossepoch=lossiter/datasetlen
        accepoch=correctruns.double()/datasetlen
        print(f'Loss:{lossepoch:.3f} Accuracy:{accepoch:.3f}')
    return model
if __name__=='__main__':
    model=train_model(model,criterion,optimizer,num_epochs=15)
    torch.save(model.state_dict(),'weights.pth')

