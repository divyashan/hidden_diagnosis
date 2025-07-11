# Standard libraries
import warnings
from timeit import default_timer as timer
from operator import __add__

# Data processing and ML libraries
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
import sys

sys.path.append('../src/')
# Custom imports
from utils import restrict_GPU_pytorch
from ecg_augs import (
    baseline_wander_mag,
    gaussian_noise_mag, 
    rand_crop_mag
)

# Configure warnings and GPU
warnings.filterwarnings('ignore', category=FutureWarning)
restrict_GPU_pytorch('0')
torch.cuda.set_device(0)

# ## Training the Network
def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=5,
          n_epochs=20,
          print_every=2,
          augmentations_on=False,
          expt_config=None,
          demographics_only=False):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    history = []
    device = torch.device("cuda:{}".format(0))
   
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    if demographics_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.dense1.parameters():
            param.requires_grad = True
        for param in model.dense2.parameters():
            param.requires_grad = True

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0
        running_corrects=0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            target = target.to(device=device, dtype=torch.float)
            cleaned_targets = torch.nan_to_num(target, nan=0)

            if expt_config['additional_features']:
                data_size = data[0].size(0)
                data = (data[0].to(device=device, dtype=torch.float), data[1].to(device=device, dtype=torch.float))
                cleaned_data = (torch.nan_to_num(data[0], nan=0), torch.nan_to_num(data[1], nan=0))
            else:
                data_size = data.size(0)
                data = data.to(device=device, dtype=torch.float)
                cleaned_data = torch.nan_to_num(data, nan=0)

            model = model.cuda()
            # Clear gradients
            optimizer.zero_grad()

            input_ecg_data = cleaned_data
            if expt_config['additional_features']:
                input_ecg_data = cleaned_data[0]
            if np.random.random() < .5 and augmentations_on:
                input_ecg_data = baseline_wander_mag(input_ecg_data)  

            if np.random.random() < .5 and augmentations_on:
                input_ecg_data = gaussian_noise_mag(input_ecg_data)

            if np.random.random() < .25 and augmentations_on:
                input_ecg_data = rand_crop_mag(input_ecg_data) 
            

            if expt_config['additional_features']:
                cleaned_data = (input_ecg_data, cleaned_data[1])
            else:
                cleaned_data = input_ecg_data
            # Predicted outputs are log probabilities
            output = model(cleaned_data)
            # Loss and backpropagation of gradients
            if expt_config['arch'].startswith('PreOpNet') or expt_config['arch'].startswith('Net1D'):
                loss = criterion(output, cleaned_targets.unsqueeze(1))
            else:
                loss = criterion(output, cleaned_targets)

            loss.backward()

            # Update the parameters
            optimizer.step()

            # Calculate accuracy by finding max log probability
            pred = torch.round(torch.sigmoid(output))
            # print("Predictions:",pred)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch


            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data_size
            train_acc += accuracy.item() * data_size

            
            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        model.epochs += 1

        # # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            pred_probs = []
            val_labels = []
            for data, target in valid_loader:
                # Tensors to gpu
                target = target.to(device=device, dtype=torch.float)
                cleaned_targets = torch.nan_to_num(target, nan=0)

                if expt_config['additional_features']:
                    data_size = data[0].size(0)
                    # keep_idxs = [x.shape == (12, 2500) for x in data[0]]
                    data = (data[0].to(device=device, dtype=torch.float), data[1].to(device=device, dtype=torch.float))
                    cleaned_data = (torch.nan_to_num(data[0], nan=0), torch.nan_to_num(data[1], nan=0))
                else:
                    data_size = data.size(0)
                    data = data.to(device=device, dtype=torch.float)
                    cleaned_data = torch.nan_to_num(data, nan=0)


                # Forward pass
                output = model(cleaned_data)
                # Validation loss
                # NOTE: The criterion looks ok but we need to sit and do some debugging with the real data.
                # I can't say by just looking at it. In practice, we should see what are the outputs.
                if expt_config['arch'].startswith('PreOpNet') or expt_config['arch'].startswith('Net1D'):
                    loss = criterion(output, cleaned_targets.unsqueeze(1))
                else:
                    loss = criterion(output, cleaned_targets)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data_size

                # Calculate validation accuracy
                pred_prob = torch.sigmoid(output)
                pred = torch.round(torch.sigmoid(output))
                running_corrects += torch.sum(pred == target.data)

                # print(running_corrects,len(valid_loader.dataset))
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data_size

                pred_probs.append(pred_prob.detach())
                val_labels.append(target.data)

            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader.dataset)
            valid_acc = valid_acc / len(valid_loader.dataset)
            valid_auc = roc_auc_score(torch.cat(val_labels).cpu(), torch.cat(pred_probs).cpu())


            history.append([train_loss, valid_loss, train_acc, valid_acc, valid_auc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                # print("epoch_acc",epoch_acc)
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )
                print(
                    f'\t\tValidation AUC: {valid_auc:.2f}'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                torch.save(model.state_dict(), save_file_name)
                best_epoch = epoch


           
            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                #Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc', 'valid_auc'
                        ])
                    return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print("\n\n----------------\n\n")
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'valid_auc'])
    return model, history
    
def eval(model, dataloader):
    model.eval()
    device = torch.device("cuda:{}".format(0))
    all_pred_prob = []
    all_pred = []
    all_labels = []
    for data, target in tqdm(dataloader):
        if len(data) == 2:
            data = (data[0].to(device=device, dtype=torch.float), data[1].to(device=device, dtype=torch.float))
            cleaned_data = (torch.nan_to_num(data[0], nan=0), torch.nan_to_num(data[1], nan=0))
        else:
            data = data.to(device=device, dtype=torch.float)
            cleaned_data = torch.nan_to_num(data, nan=0)            
        
        cleaned_targets = torch.nan_to_num(target, nan=0)
        output = model(cleaned_data)
        pred_prob = torch.sigmoid(output)
        pred = torch.round(torch.sigmoid(output))
        all_pred_prob.append(pred_prob.detach())
        all_pred.append(pred.detach())
        all_labels.append(cleaned_targets.detach())

    # Calculate AUC
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_pred = torch.cat(all_pred).cpu().numpy()
    all_pred_prob = torch.cat(all_pred_prob).cpu().numpy()
    acc = np.mean(all_labels == all_pred)
    sensitivity = np.mean(all_pred[all_labels == 1])
    specificity = 1 - np.mean(all_pred[all_labels == 0])
    auc = roc_auc_score(all_labels, all_pred_prob)
    return acc, auc, sensitivity, specificity


