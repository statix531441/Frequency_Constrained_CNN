import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ECG(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        signal, label = self.df.iloc[index, :-1], self.df.iloc[index, -1]
        signal = torch.as_tensor(signal)
        label = torch.as_tensor(label)
        return signal.view(1,140).float(), label.long()
    

class MNIST_1D(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        pass

    
Datasets = {
    'ECG': ECG,
    'MNIST_1D': MNIST_1D,
}

if __name__ == "__main__":
    import pandas as pd
    from options import Options

    opt = Options(dataset='ECG', model='SmoothCNN', tag='DELETE')
    df = pd.read_csv(f"original/ecg.csv", header=None)

    Dataset = Datasets[opt.dataset]
    test_set = Dataset(df)

    X, y = test_set[10]
    print(X.shape, y)