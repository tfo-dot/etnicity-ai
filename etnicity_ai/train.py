import torch
import os

from . import data, metrics, utils

from tqdm import tqdm
from pathlib import Path

def train_model(model, criterion, optimizer, scheduler_lr, device, loaders, tensor_writer, mappings, class_dicts, class_weights, n_epochs=25):
    """Fine-tune the model and returns it
    Args:
        model: The model to be trained.
        criterion: The loss function to be used.
        optimizer: The optimizer to be used.
        scheduler_lr: The learning rate scheduler to be used.
        device: The device to be used.
        loaders: A dictionary containing the train and test data loaders.
        tensor_writer: A TensorBoard writer.
        mappings: A dictionary mapping labels to their corresponding indices.
        n_epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """
    # unpack_loss_coefs
    race_weight: float = class_weights['race']
    # Whether to use learning rate scheduler
    use_lr_scheduler = False

    # Train and validation data pytorch loaders 
    train_loader, test_loader = loaders
    # Unpack task dicts

    race_dict = class_dicts['race']

    # In the custom implementation of the pytorch dataset train_mode attribute was added for more convenient observance of the image
    data.check_train_mode_off(train_loader, 'train')
    data.check_train_mode_off(test_loader, 'test')

    # Flag that shows whether CCE or BCE is used during training
    cce_mark = type(criterion) == torch.nn.modules.loss.CrossEntropyLoss

    # Initializing of minimal value of validation loss to compare in following code
    valid_loss_min = torch.inf

    # starts training epochs[default = 25]
    for epoch in range(1, n_epochs):
        # Define train and each ANN head losses to compare if it is decreasing during training process
        train_loss = .0
        age_train_loss, race_train_loss, gender_train_loss = .0, .0, .0

        model.train()

        # whether learning rate scheduler is used make the step of it or not
        if use_lr_scheduler:
            scheduler_lr.step()
        
        # ground truths and predictions of the each head of the ANN
        all_race_preds = []
        all_race_labels = []

        # As far as loaders have mini-batch feature and provide samples in that amounts, so iterate over these mini-batches 
        for sample_batched in tqdm(train_loader):
            # read every element of the sample and send it to device[GPU as it is expected]
            image, race = sample_batched['image'].float().to(device), sample_batched['race'].to(device)

            # get model's predictions
            output = model(image)
            # predict age, gender, race labels
            race_pred = output['race_pred'].to(device)
            
            # calculate loss[CCE or BCE]
            if cce_mark:
                race_loss = criterion(race_pred, race)
            else:
                # If it is BCE firstly convert to the one-hot encoded vector
                race_dummy = data.get_dummy(race_dict, race, device)
                
                # Then calculate the loss
                race_loss = criterion(race_pred, race_dummy)
                
            # total loss and back propagation
            loss = race_loss * race_weight

            # add the loss of age, race, gender heads to their total losses of the epoch
            race_train_loss += race_loss
            train_loss += loss 

            # Clear the gradients of all parameters
            optimizer.zero_grad()
            # Compute the gradients of the loss function with respect to the model's parameters
            loss.backward()
            # optimization step
            optimizer.step()

            # Get Race prediction
            _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)
            
            # Append computed above predictions to the vector for the following comparing
            all_race_preds.extend([i.item() for i in race_predicted])

            # Append true labels of the dataset to the vector for the following comparing 
            all_race_labels.extend([i.item() for i in race])

        # Every epoch log metrics and losses to the tensorboard to track whether the model is training or not
        
        losses = {'loss' : train_loss,'race' : race_train_loss}
        losses = {k : v / len(train_loader) for (k, v) in losses.items()}
        labels = {'race' : all_race_labels}
        preds = {'race' : all_race_preds}

        print('Training step:')
        tensor_writer = metrics.log_tensorboard(tensor_writer, losses, labels, preds, mappings, epoch, mode='Train')


        # Start evaluation step
        model.eval()

        # Don't compute gradients during this step
        with torch.no_grad():
            # Define vectors of true labels and predictions, current validation loss and each head's loss as like as above
            # for accuracy, precision, recall, f1 etc. comparing
            valid_race_preds = []
            valid_race_labels = []

            valid_loss = .0
            valid_race_loss = .0

            # Iterate over the batches of the validation dataloader
            for sample_batched in tqdm(test_loader):
                # read every element of the sample and send it to device[GPU as it is expected]
                image, = sample_batched['age'].to(device)
                race = sample_batched['race'].to(device)

                # get model's predictions
                output = model(image)

                # predict race label
                race_pred = output['race_pred'].to(device)

                # calculate loss[CCE or BCE]
                if cce_mark:
                    race_loss = criterion(race_pred, race)
                else:
                    # If it is BCE firstly convert to the one-hot encoded vector
                    race_dummy = data.get_dummy(race_dict, race, device)

                    # Then calculate the loss
                    race_loss = criterion(race_pred, race_dummy)

                # total loss and back propagation
                loss = race_loss * race_weight

                # add the loss of age, race, gender heads to their total losses of the epoch
                valid_loss += loss.item()
                valid_race_loss += race_loss

                # Get Race prediction 
                _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)

                # Append computed above predictions to the vector for the following comparing
                valid_race_preds.extend([i.item() for i in race_predicted])

                # Append true labels of the dataset to the vector for the following comparing 
                valid_race_labels.extend([i.item() for i in race])

            # Every epoch log metrics and losses to the tensorboard to track whether the model is training or not
            losses = {'loss' : valid_loss, 'race' : valid_race_loss}
            losses = {k : v / len(test_loader) for (k, v) in losses.items()}
            labels = {'race' : valid_race_labels}
            preds = {'race' : valid_race_preds}

            print('Validation step:')
            tensor_writer = metrics.log_tensorboard(tensor_writer, losses, labels, preds, mappings, epoch, mode='Validation')

        # print training/validation statistics 
        valid_loss /= len(test_loader)
        train_loss /= len(train_loader)

        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        checkpoint_dir = Path('../models/checkpoint/')
        best_model_dir = Path('../models/best_model/')

        print(f'End of Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}')
        
        # if validation loss of the current epoch is lower than previous one - save the model
        if valid_loss < valid_loss_min:
            models_path = '../models/'
            if not os.path.exists(models_path):
                for path in ('../models/', checkpoint_dir, best_model_dir):
                    os.mkdir(path)
            utils.checkpoint(model_state, True, checkpoint_dir, best_model_dir)
            print(f'Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...')
            valid_loss_min = valid_loss
        
        # Do the checkpoint for the model every epoch
        utils.checkpoint(model_state, False, checkpoint_dir, best_model_dir)

    # return trained model
    return model

