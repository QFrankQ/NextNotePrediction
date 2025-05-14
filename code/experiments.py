import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from load_data import CustomDataset, load_data, load_test_data
import torch.optim as optim
from model import MixOfModels
from torch import nn
import Math
import argparse

#Load the first training song
def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size')
    # hint_cells_indices             # sol_cells_indices
    # correct_sol_cells_indices      # error_sol_cells_indices
    # masked_error_sol_cells_indices # unmask_error_sol_cells_indices
    
    parser.add_argument('--lr', default=0.001, type=int,
                        help='learning rate')
    parser.add_argument('--num-epochs', default=20, type=int,)
    
    parser.add_argument('--encoder_num_layers', default=2, type=int)
    parser.add_argument('--decoder_num_layers', default=2, type=int)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('pretrained', default=False, type=bool)
    
    return parser


def train(model, training_dataloader, validation_dataloader, optimizer, args):
    train_loss = []
    # Start loop
    best_val_loss = Math.inf
    device = args.gpu_id
    num_epochs = args.num_epochs
    counter = 0 
    for epoch in range(num_epochs): #(loop for every epoch)

        model.train()    # training mode
        running_loss = 0.0   

        for (X, Y) in training_dataloader: # load a batch data of images

            #Move batch to device if needed
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad() #Zero the gradient
            output = model.forward(X)  #Compute predicted probabilities          
            time_mu_std, duration_mu_std, note_dist, volume_mu_std = output[:,0:2], output[:,2:4], output[:,4:132], output[:,132:134]
            note_dist=torch.clip(note_dist,0.01,0.99) #Clip the probabilities to avoid log(0)
            
            GaussianNLL = nn.GaussianNLLLoss()
            CrossEntropy = nn.CrossEntropyLoss()
            input = torch.cat(time_mu_std[:,0], duration_mu_std[:,0], volume_mu_std[:,0], dim=1)
            target = torch.cat(Y[:,0], Y[:,1], Y[:,3], dim=1)
            var = torch.cat(time_mu_std[:,1], duration_mu_std[:,1], volume_mu_std[:,1], dim=1)
            GaussianNLL_loss = GaussianNLL(input, target, var)
            note_CrossEntropy_loss = CrossEntropy(note_dist, Y[:,2])
            loss = GaussianNLL_loss + note_CrossEntropy_loss
            loss.backward()       #Compute the gradient of the loss
            optimizer.step()      #Take a step
            # update running loss and error
            running_loss  += loss.item() * X.shape[0]

        #Compute loss for the epoch
        epoch_loss  = running_loss /  len(training_dataloader.dataset)

        # Append result
        train_loss.append(epoch_loss)

        # Print progress
        print('[Train #{}] Loss: {:.8f}'.format(epoch+1, epoch_loss))
        
        if epoch > 15:
            val_loss = validate_model(model, validation_dataloader)
            print('[Val #{}] Loss: {:.8f} '.format(epoch+1, val_loss))
            if val_loss >= best_val_loss:
                counter += 1
            if counter >= 5:
                break
            best_val_loss = val_loss
    torch.save(model.state_dict(), f'model/checkpoint_last.pth')
    return model
    
def validate_model(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (X,Y) in val_loader:
            # model = model.eval()
            X = X.to(device)
            Y = Y.to(device)
            output = model.forward(X)  #Compute predicted probabilities          
            time_mu_std, duration_mu_std, note_dist, volume_mu_std = output[:,0:2], output[:,2:4], output[:,4:132], output[:,132:134]
            note_dist=torch.clip(note_dist,0.01,0.99) #Clip the probabilities to avoid log(0)
            
            GaussianNLL = nn.GaussianNLLLoss()
            CrossEntropy = nn.CrossEntropyLoss()
            input = torch.cat(time_mu_std[:,0], duration_mu_std[:,0], volume_mu_std[:,0], dim=1)
            target = torch.cat(Y[:,0], Y[:,1], Y[:,3], dim=1)
            var = torch.cat(time_mu_std[:,1], duration_mu_std[:,1], volume_mu_std[:,1], dim=1)
            GaussianNLL_loss = GaussianNLL(input, target, var)
            note_CrossEntropy_loss = CrossEntropy(note_dist, Y[:,2])
            loss = GaussianNLL_loss + note_CrossEntropy_loss
            running_loss  += loss.item() * X.shape[0]
    return loss


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    
    files=glob.glob("../data/songs/train/*/*/*/*.pt")
    X = torch.load(files[0])

    train_X, train_Y, val_X, val_Y = load_data()
    

    # Initialize dataset and dataloader
    training_dataset = CustomDataset(train_X, train_Y)
    training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    validation_dataset = CustomDataset(val_X, val_Y)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)



    model = MixOfModels(4, args.d_model, args.encoder_num_layer, args.decoder_num_layer).to(args.gpu_id)
    lr         = 0.001
    num_epochs = 20
    train_loss = []

    optimizer  = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-1)
    if args.pretrained:
        model.load_state_dict(torch.load('model/checkpoint_last.pth'))
    else:
        model = train(model, training_dataloader, validation_dataloader, optimizer, args)
    #random integer of shape (1,64, 4)
    random_input = torch.randint(0, 128, (1,64,4))
    
# #Load the test contexts
# Xtest = torch.load("../data/songs/test.pt")
# print(Xtest[0,:5,:])

#Generate and save a test predictions file
# Ntest = Xtest.shape[0]
# row   = torch.tensor([[0,1,0,1,*([np.log(1/128.0)]*128),0,1]])
# Yhat  = row.repeat(Ntest,1)
# torch.save(Yhat, "note_predictions.pt")
# print(Yhat.shape)