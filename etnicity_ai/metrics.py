import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from tqdm import tqdm

def create_cm_df(cf, classes):
    """
    Creates a confusion matrix DataFrame.

    Args:
        cf: A confusion matrix.
        classes: A list of the classes in the confusion matrix.

    Returns:
        A confusion matrix DataFrame.
    """
    return pd.DataFrame(cf / np.sum(cf, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])

def create_cfs(ages, genders, races, mappings):
    """
    Creates a confusion matrix plot for each class.

    Args:
        ages: A dictionary of age labels and predictions.
        genders: A dictionary of gender labels and predictions.
        races: A dictionary of race labels and predictions.
        mappings: A dictionary mapping labels to their corresponding indices.

    Returns:
        A figure containing the confusion matrix plots.
    """
    race_cf = confusion_matrix(races['true'], races['pred'])

    race_df = create_cm_df(race_cf, mappings['race_map'])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (24, 5))
    sns.heatmap(race_df, annot=True, ax=axes[2])
    plt.close()
    return fig

def build_confusion_matrix(model, loader, mappings, device):
    """Builds a confusion matrix for the given model, loader, mappings, and device.

    Args:
        model: The model to evaluate.
        loader: The loader to use to load the data.
        mappings: A dictionary of mappings from labels to indices.
        device: The device to use.

    Returns:
        A tuple of three dictionaries: ages, genders, and races. Each dictionary contains two lists: pred and true.
    """
    races = {'pred' : [], 'true' : []}
    # Perform the full cycle of the iterating over the whole test dataloder and append predictions with labels to the vectors
    with torch.no_grad():   
          
        for batch in tqdm(loader):
            image, race = batch['image'].to(device), batch['race'].to(device)
            
            output = model(image)

            # Race accuracy
            _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)
            
            races['pred'].extend(race_predicted.cpu())

            races['true'].extend(race.cpu())

    race_cf = confusion_matrix(races['true'], races['pred'])

    race_df = create_cm_df(race_cf, mappings['race_map'])

    # Plot confusion matrices for every head of the provided Neural Network
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (24, 5))
    sns.heatmap(race_df, annot=True, ax=axes[2])
    plt.show()


    return races

def calc_measures(model, loader, device):
    """Calculates the accuracy, precision, recall, and f1-score for the given model and loader.

    Args:
        model: The model to evaluate.
        loader: The loader to use to load the data.

    Returns:
        A Pandas DataFrame containing the accuracy, precision, recall, and f1-score for each attribute (age, gender, and race).
    """
    # Initialize the dictionaries to store the predictions and labels.
    races = {'pred' : [], 'true' : [] }

    # Create a dictionary to store the measures.
    measures = {}

    # Iterate over the loader.
    with torch.no_grad():   
          
        for batch in tqdm(loader):
            # Move the images and labels to the device.
            image, race = batch['image'].to(device), batch['race'].to(device)
            
            # Make a prediction.
            output = model(image)
            # Get the predicted and true labels.
            _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)
            
            # Add the predictions and labels to the dictionaries.
            races['pred'].extend(race_predicted.cpu())

            races['true'].extend(race.cpu())

    # Calculate the accuracy, precision, recall, and f1-score for each attribute.
    measures['race'] = [accuracy_score(races['true'], races['pred']), precision_score(races['true'], races['pred'], average='macro'), 
                        recall_score(races['true'], races['pred'], average='macro'), f1_score(races['true'], races['pred'], average='macro')]
    # Return a Pandas DataFrame containing the measures.
    return pd.DataFrame(measures, index=['accuracy', 'precision', 'recall', 'f1']).T

def log_tensorboard(board, losses, labels, preds, mappings, epoch, mode='Train'):

    loss = losses['loss']

    race_loss = losses['race']
    race_labels = labels['race']
    race_preds = preds['race']
    
    accuracy_str += f'Race accuracy = {accuracy_score(race_labels, race_preds)}\t'
    
    print(f'Epoch = {epoch}, Loss = {loss}')
    print(accuracy_str)
    
    board.add_scalar(f'{mode} Loss', loss, epoch)
    board.add_scalar(f'Race {mode} Loss', race_loss, epoch)
    
    board.add_scalar(f'Race {mode} accuracy', accuracy_score(race_labels, race_preds), epoch)
    
    board.add_scalar(f'Race {mode} f1', f1_score(race_labels, race_preds, average='macro'), epoch)
    
    board.add_scalar(f'Race {mode} precision', precision_score(race_labels, race_preds, average='macro'), epoch)
    
    board.add_scalar(f'Race {mode} recall', recall_score(race_labels, race_preds, average='macro'), epoch)
    
    # Create confusion matrix each epoch and label it the same way for more ease comparing during epochs
    board.add_figure(f'{mode.lower()}_fairface_cm', create_cfs(races={'true' :   race_labels, 'pred' : race_preds}, 
                        mappings=mappings), global_step=epoch)
    
    return board