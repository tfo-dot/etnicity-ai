import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from tqdm import tqdm

def create_cm_df(cf, classes):
    return pd.DataFrame(cf / np.sum(cf, axis=1, keepdims=True), index=classes, columns=classes)

def create_cfs(races, mappings):
    race_cf = confusion_matrix(races['true'], races['pred'])
    race_df = create_cm_df(race_cf, mappings['race_map'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(race_df, annot=True, ax=ax)
    plt.close()
    return fig

def build_confusion_matrix(model, loader, mappings, device):
    races = {'pred': [], 'true': []}
    with torch.no_grad():
        for batch in tqdm(loader):
            image, race = batch['image'].to(device), batch['race'].to(device)
            output = model(image)
            _, race_predicted = torch.max(output['race_pred'].data, 1)
            races['pred'].extend(race_predicted.cpu().numpy())
            races['true'].extend(race.cpu().numpy())
    
    race_cf = confusion_matrix(races['true'], races['pred'])
    race_df = create_cm_df(race_cf, mappings['race_map'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(race_df, annot=True, ax=ax)
    plt.show()
    return races

def calc_measures(model, loader, device):
    races = {'pred': [], 'true': []}
    with torch.no_grad():
        for batch in tqdm(loader):
            image, race = batch['image'].to(device), batch['race'].to(device)
            output = model(image)
            _, race_predicted = torch.max(output['race_pred'].data, 1)
            races['pred'].extend(race_predicted.cpu().numpy())
            races['true'].extend(race.cpu().numpy())
    
    measures = {
        'race': [
            accuracy_score(races['true'], races['pred']),
            precision_score(races['true'], races['pred'], average='macro', zero_division=0),
            recall_score(races['true'], races['pred'], average='macro', zero_division=0),
            f1_score(races['true'], races['pred'], average='macro', zero_division=0)
        ]
    }
    return pd.DataFrame(measures, index=['accuracy', 'precision', 'recall', 'f1']).T

def log_tensorboard(board, losses, labels, preds, mappings, epoch, mode='Train'):
    loss, race_loss = losses['loss'], losses['race']
    race_labels, race_preds = labels['race'], preds['race']
    
    print(f'Epoch = {epoch}, Loss = {loss}')
    print(f'Race accuracy = {accuracy_score(race_labels, race_preds)}')
    
    board.add_scalar(f'{mode} Loss', loss, epoch)
    board.add_scalar(f'Race {mode} Loss', race_loss, epoch)
    board.add_scalar(f'Race {mode} accuracy', accuracy_score(race_labels, race_preds), epoch)
    board.add_scalar(f'Race {mode} f1', f1_score(race_labels, race_preds, average='macro', zero_division=0), epoch)
    board.add_scalar(f'Race {mode} precision', precision_score(race_labels, race_preds, average='macro', zero_division=0), epoch)
    board.add_scalar(f'Race {mode} recall', recall_score(race_labels, race_preds, average='macro', zero_division=0), epoch)
    
    board.add_figure(f'{mode.lower()}_fairface_cm', create_cfs({'true': race_labels, 'pred': race_preds}, mappings), epoch)
    return board
