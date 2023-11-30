import torch
import torch.nn as nn
from torch.functional import F
from torch.fft import fft, ifft

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
    def forward(self, X):
        out = self.conv(X)
        out = F.relu(out)
        # out = F.avg_pool1d(out, kernel_size=3)
        return out
    

class CNN(nn.ModuleDict):
    def __init__(self, opt):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv_layer", nn.Conv1d(1, opt.channels[0], opt.kernel_size))
        for i, (nin, nout) in enumerate(zip(opt.channels[:-1], opt.channels[1:])):
            self.features.add_module(f"conv_layer_{i}", conv_layer(nin, nout, opt.kernel_size))
        self.features.add_module(f"batch_norm", nn.BatchNorm1d(opt.channels[-1]))
        self.classifier = nn.Linear(opt.channels[-1], opt.classes)

    def forward(self, X):
        out = self.features(X)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
class constrained_conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pct=0.1):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.masks = nn.Parameter(torch.zeros_like(self.freqs, dtype=bool), requires_grad=False)
        self.masks[:, :, :int(pct*self.masks.shape[2])] = True
        self.freqs.data *= self.masks

    def forward(self, X):
        freqs = self.freqs * self.masks
        weights = ifft(freqs).real
        out = F.conv1d(X, weights) + self.bias.view(1, -1, 1)
        out = F.relu(out)
        # out = F.avg_pool1d(out, kernel_size=2)
        return out
    
class SmoothCNN(nn.ModuleDict):
    def __init__(self, opt):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv_layer", constrained_conv_layer(1, opt.channels[0], opt.kernel_size, opt.pct))
        for i, (nin, nout) in enumerate(zip(opt.channels[:-1], opt.channels[1:])):
            self.features.add_module(f"conv_layer_{i}", constrained_conv_layer(nin, nout, opt.kernel_size, opt.pct))
        self.features.add_module(f"batch_norm", nn.BatchNorm1d(opt.channels[-1]))
        self.classifier = nn.Linear(opt.channels[-1], opt.classes)

    def forward(self, X):
        out = self.features(X)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
class SmoothCNN2(nn.ModuleDict):
    def __init__(self, opt):
        super().__init__()

    def forward(self, X):
        out = X
        return out
    

Models = {
    'CNN': CNN,
    'SmoothCNN': SmoothCNN,
    'SmoothCNN2': SmoothCNN2,
}

if __name__ == "__main__":
    from options import Options
    from models import *
    from dataset import *
    from utils import *

    opt = Options(dataset='ECG', model='SmoothCNN', tag='DELETE')
    
    model = Models['SmoothCNN'] (opt)
    test_df = pd.read_csv(f"{opt.data_folder}/test.csv", header=None)
    test_set = ECG(test_df)
    test_loader = DataLoader(test_set, batch_size=30, shuffle=True)

    X = torch.randn(30, 1, 140)
    out = model(X)
    print(out.shape)

    y_pred, y_test, test_accuracy = model.predict(test_loader)
    print(torch.sum(torch.tensor(y_pred)==torch.tensor(y_test))/len(y_test), test_accuracy)



