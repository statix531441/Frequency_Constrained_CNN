import pandas as pd
import torch
from torch.functional import F

# Loads from original and saves changes into data/
def create_split(opt):
    if opt.dataset=='ECG':
        data = pd.read_csv('original/ecg.csv', header=None)
        train_df = data.sample(frac=opt.split, random_state=42)
        test_df = data.drop(train_df.index)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        if opt.add_noise:
            test_df.loc[:, range(140)] += torch.randn(len(test_df), 140).numpy() * opt.std + opt.mean

        train_df.to_csv(f'{opt.data_folder}/train.csv', index=False, header=None)
        test_df.to_csv(f'{opt.data_folder}/test.csv', index=False, header=None)

    elif opt.dataset=='MNIST':
        pass

    return train_df, test_df

def fit(model, train_loader, lossFn, optimizer):  
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train_loss = 0
    train_accuracy = 0

    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = lossFn(pred, y)
        # print(loss.item())
        train_loss += loss.item()
        train_accuracy += torch.sum(F.softmax(pred, dim=1).argmax(axis=1) == y).item()
        loss.backward()
        optimizer.step()
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy

def predict(model, test_loader):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    y_test = []
    y_pred = []
    test_accuracy = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_test.extend(y.tolist())
            y_pred.extend(F.softmax(pred, dim=1).argmax(axis=1).tolist())
            test_accuracy += torch.sum(F.softmax(pred, dim=1).argmax(axis=1) == y).item()
    test_accuracy /= len(test_loader.dataset)
    return y_pred, y_test, test_accuracy