import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

from data import CustomDataset
from utils import Trainer


# Classifier with adapted classification head
class SentimentClassifier(nn.Module):
    def __init__(self,n_classes):
        super(SentimentClassifier, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids,attention_mask=attention_mask,return_dict = False)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)

# Function for plotting model progress over training
def plot_log(ax, train_value, val_value, label):
    sns.lineplot(y=train_value, x=range(1, hyperparameter['epochs'] + 1), ax=ax, label=f'train_{label}')
    sns.lineplot(y=val_value, x=range(1, hyperparameter['epochs'] + 1), ax=ax, label=f'val_{label}')
    ax.set_title(f'Change of {label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)

if __name__ == '__main__':

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    data = pd.read_csv('./data.csv') # Load dataset
    data.drop_duplicates(inplace=True)  # Drop duplicates
    data.dropna(inplace=True)  # Handle missing values (if any)

    # One hot encoding of the target variable
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data.Sentiment)

    # Configuration of the Model and Hyperparameters
    model_name = 'bert-base-uncased'
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    hyperparameter = {
        'max_len': 50,
        'batch_size': 16,
        'device': torch.device('cuda'),
        'lr_rate': 2e-5,
        'epochs': 3,
        'num_classes': 3,
    }

    # Train, val, test split (80,10,10)
    X_train, X_val_test, y_train, y_val_test = train_test_split(data.Sentence.values, labels, test_size=.2,
                                                                random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5, random_state=random_seed)

    train_dataset = CustomDataset(X_train, y_train, tokenizer, hyperparameter['max_len'])
    train_loader = DataLoader(train_dataset,
                              batch_size=hyperparameter['batch_size'],
                              num_workers=2)

    val_dataset = CustomDataset(X_val, y_val, tokenizer, hyperparameter['max_len'])
    val_loader = DataLoader(val_dataset,
                            batch_size=hyperparameter['batch_size'],
                            num_workers=2)

    test_dataset = CustomDataset(X_test, y_test, tokenizer, hyperparameter['max_len'])
    test_loader = DataLoader(test_dataset,
                             batch_size=hyperparameter['batch_size'],
                             num_workers=2)

    # Start Finetuning
    model = SentimentClassifier(hyperparameter['num_classes'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameter['lr_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    trainer = Trainer(model=model,
                      crit=criterion,
                      optim=optimizer,
                      train_dl=train_loader,
                      val_test_dl=val_loader,
                      device='cuda',
                      early_stopping_patience=30,
                      scheduler=scheduler)

    # Call fit on trainer
    log = trainer.fit(hyperparameter['epochs'])

    # Model Evaluation
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    plot_log(axs[0], log.train_loss, log.val_loss, 'Loss')
    plot_log(axs[1], log.train_accuracy, log.val_accuracy, 'Accuracy')
    plt.show()

    device = hyperparameter['device']
    model = SentimentClassifier(hyperparameter['num_classes'])
    model.load_state_dict(torch.load('./best.pth'))
    model.to(device)
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            targets = batch['targets']

            input_ids = input_ids.to(device, non_blocking=True).long()
            attention_mask = attention_mask.to(device, non_blocking=True).long()
            targets = targets.to(device, non_blocking=True).long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            preds_list = [pred.item() for pred in preds]
            labels_list = [label.item() for label in targets]

            labels.extend(labels_list)
            predictions.extend(preds_list)

    # visualize predictions
    sns.set_theme(style='whitegrid', rc={'figure.figsize': (8, 6)})
    cf_matrix = confusion_matrix(labels, predictions)
    sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt="g", xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)

    label_preds = encoder.inverse_transform(predictions)
    label_targets = encoder.inverse_transform(labels)

    print(classification_report(label_targets, label_preds))