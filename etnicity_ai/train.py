import torch
import os
from pathlib import Path
from tqdm import tqdm

from . import data, metrics, utils

def train_model(model, criterion, optimizer, scheduler_lr, device, loaders, tensor_writer, mappings, class_dicts, n_epochs=25):
    """Fine-tune the model and return it."""
    
    # Enable learning rate scheduler if provided
    use_lr_scheduler = scheduler_lr is not None
    
    # Train and validation data loaders 
    train_loader, test_loader = loaders
    race_dict = class_dicts['race']
    
    # Ensure dataset is in training mode
    data.check_train_mode_off(train_loader, 'train')
    data.check_train_mode_off(test_loader, 'test')
    
    # Check if using CrossEntropyLoss
    cce_mark = isinstance(criterion, torch.nn.CrossEntropyLoss)
    
    # Initialize minimal validation loss for early stopping
    valid_loss_min = torch.inf
    
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        model.train()
        
        # Track predictions and ground truths
        all_race_preds, all_race_labels = [], []

        for sample_batched in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            image, race = sample_batched['image'].float().to(device), sample_batched['race'].to(device)
            
            optimizer.zero_grad()
            model.zero_grad()
            
            output = model(image)
            race_pred = output['race_pred']
            
            # Compute loss
            if cce_mark:
                race_loss = criterion(race_pred, race)
            else:
                race_dummy = data.get_dummy(race_dict, race, device)
                race_loss = criterion(race_pred, race_dummy)
            
            race_loss.backward()
            optimizer.step()
            
            train_loss += race_loss.item()
            
            _, race_predicted = torch.max(race_pred, 1)  # No need for softmax
            all_race_preds.extend(race_predicted.cpu().tolist())
            all_race_labels.extend(race.cpu().tolist())
        
        # Log training metrics
        losses = {'loss': train_loss / len(train_loader), 'race': race_loss.item()}
        labels, preds = {'race': all_race_labels}, {'race': all_race_preds}
        print('Training step:')
        tensor_writer = metrics.log_tensorboard(tensor_writer, losses, labels, preds, mappings, epoch, mode='Train')
        
        # Validation
        model.eval()
        valid_loss, valid_race_preds, valid_race_labels = 0.0, [], []
        
        with torch.no_grad():
            for sample_batched in tqdm(test_loader, desc=f"Epoch {epoch} - Validation"):
                image, race = sample_batched['image'].float().to(device), sample_batched['race'].to(device)
                
                output = model(image)
                race_pred = output['race_pred']
                
                if cce_mark:
                    race_loss = criterion(race_pred, race)
                else:
                    race_dummy = data.get_dummy(race_dict, race, device)
                    race_loss = criterion(race_pred, race_dummy)
                
                valid_loss += race_loss.item()
                
                _, race_predicted = torch.max(race_pred, 1)
                valid_race_preds.extend(race_predicted.cpu().tolist())
                valid_race_labels.extend(race.cpu().tolist())
            
        # Log validation metrics
        losses = {'loss': valid_loss / len(test_loader), 'race': race_loss.item()}
        labels, preds = {'race': valid_race_labels}, {'race': valid_race_preds}
        print('Validation step:')
        tensor_writer = metrics.log_tensorboard(tensor_writer, losses, labels, preds, mappings, epoch, mode='Validation')
        
        print(f'End of Epoch {epoch} \t Training Loss: {train_loss / len(train_loader):.4f} \t Validation Loss: {valid_loss / len(test_loader):.4f}')
        
        # Adjust learning rate based on validation loss
        if use_lr_scheduler:
            scheduler_lr.step(valid_loss)
        
        # Save model if validation loss decreases
        checkpoint_dir, best_model_dir = Path('../models/checkpoint/'), Path('../models/best_model/')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        
        model_state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        
        if valid_loss < valid_loss_min:
            utils.checkpoint(model_state, True, checkpoint_dir, best_model_dir)
            print(f'Validation loss decreased ({valid_loss_min:.4f} --> {valid_loss:.4f}). Saving model ...')
            valid_loss_min = valid_loss
        
        # Save checkpoint every epoch
        utils.checkpoint(model_state, False, checkpoint_dir, best_model_dir)
        
    return model
