import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from nn import bag_of_words, tokenize, stem
from brain import NeuralNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

with open('intents.json','r') as f:
    intents=json.load(f)

all_words=[]
tags =[]
xy= []

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))


ignore_words=[',','?','.','/','!']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))

tags = sorted(set(tags))

x_train=[]
y_train=[]

for (pattern_sentence,tag)in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)

    label=tags.index(tag)
    y_train.append(label)


x_train=np.array(x_train)
y_train=np.array(y_train)

num_apochs=1000
batch_size=8
lerarning_rate=0.001
print(len(x_train[0]))
input_size=len(x_train[0])
hidden_size=8
output_size=len(tags)
print(len(tags))
    
print("training the Model")

class ChatDataset(Dataset):
  
   def __init__(self):
       self.n_samples = len(x_train)
       self.x_data=x_train
       self.y_data=y_train

   def __getitem__(self,index):
       return self.x_data[index],self.y_data[index]

   def __len__(self):
       return self.n_samples

dataset = ChatDataset()

train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device=device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lerarning_rate)

for epoch in range(num_apochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(dtype=torch.long).to(device)
        outputs=model(words)
        loss=criterion(outputs,labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1) % 100==0:
        print(f'Epoch [{epoch+1}/{num_apochs}], loss: {loss.item():.4f}')

print(f'final Loss: {loss.item():.4f}')


data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags
}

FILE="TrainData.pth"
torch.save(data,FILE)




model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(output_size, output_size)

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)

        for i, j in zip(predicted, labels):
            confusion_matrix[i, j] += 1

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    

    print(f'Accuracy: {accuracy:.2f}%')


    confusion_matrix = confusion_matrix.cpu().numpy()
with torch.no_grad():
    y_true = []
    y_pred = []

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f'F1 score: {f1_score:.2f}')

print(f"Training Complete file saved to{FILE}")
