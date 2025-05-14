import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


#Load the first training song
    
def generate_training_dataset(dataset, context_window):        
    training_instances = []
    training_labels = []
    for data in dataset:
        data_len = data.shape[0]
        
        i = torch.randint(0,int(data_len/2)-context_window-10,(1,)).item()
        
        for j in range(i, i+8):
            training_instance = data[j:j+context_window,:]
            training_label = data[j+context_window+1,:]
            training_instances.append(training_instance)
            training_labels.append(training_label)
        for _ in range(20):
            i = torch.randint(0,int(data_len/2)-context_window-10,(1,)).item()
            training_instance = data[j:j+context_window,:]
            training_label = data[j+context_window+1,:]
            training_instances.append(training_instance)
            training_labels.append(training_label)
    training_instances = torch.tensor(np.array(training_instances))
    training_labels = torch.tensor(np.array(training_labels)).unsqueeze(1)
    training_dataset = torch.cat((training_instances, training_labels), dim=1)
    print(training_dataset.size())
    torch.save(training_dataset, "../data/training_dataset.pt")
    print(f"Training dataset saved, size: {training_dataset.size()}")
    
        

def generate_validation_dataset(dataset, context_window):
    validation_instances = []
    validation_labels = []
    for data in dataset:
        data_len = data.shape[0]
        for _ in range(16):
            j = torch.randint(int(data_len/2),data_len-context_window-10,(1,)).item()
            validation_instance = data[j:j+context_window,:]
            validation_label = data[j+context_window+1,:]
            validation_instances.append(validation_instance)
            validation_labels.append(validation_label)
    
    validation_instances = torch.tensor(np.array(validation_instances))
    validation_labels = torch.tensor(np.array(validation_labels)).unsqueeze(1)
    validation_dataset = torch.cat((validation_instances, validation_labels), dim=1)
    
    torch.save(validation_dataset, "../data/validation_dataset.pt")
    print(f"Validation dataset saved, size: {validation_dataset.size()}")
    
def generate_train_val_dataset(context_window):
    files=glob.glob("../data/songs/train/*/*/*/*.pt")
    dataset = []
    for file in files:
        dataset.append(torch.load(file))
    generate_training_dataset(dataset, context_window)
    generate_validation_dataset(dataset, context_window)
    
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)  # Apply transformation to x if needed

        return x, y

def load_data():
    #load training_dataset.pt
    training_dataset = torch.load("../data/training_dataset.pt")
    #load validation_dataset.pt
    validation_dataset = torch.load("../data/validation_dataset.pt")
    train_X = training_dataset[:,:-1]
    train_Y = training_dataset[:,-1]
    val_X = validation_dataset[:,:-1]
    val_Y = validation_dataset[:,-1]
    return train_X, train_Y, val_X, val_Y

if __name__ == "__main__":
    
    context_window = 64
    # generate_train_val_dataset(context_window)
    
    
    


# train_size = int(0.8 * len(files))
# val_size = len(files) - train_size


# train_data, val_data = torch.utils.data.random_split(files, [train_size, val_size])
# print(X[:5,:])

#Load the test contexts
# Xtest = torch.load("../data/songs/test.pt")
# print(Xtest[0,:5,:])