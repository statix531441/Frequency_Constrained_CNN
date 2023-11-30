import json
import os
class Options:
    def __init__(self, dataset='ECG', model="SmoothCNN", tag="NoisyTest"):
    
        # Initialize
        self.dataset = dataset
        self.model = model
        self.tag = tag
        self.split = 0.04

        # Train and test options
        self.epochs = 200
        self.batch_size = 30
        self.lr = 1e-4

        # Activate Dataset Options
        if dataset == 'ECG':
            self.classes = 2
            self.ECG()
        
        # Activate Model Options
        if model == 'CNN':
            self.CNN()
        elif model == 'SmoothCNN':
            self.SmoothCNN()

    # Dataset Options
    def ECG(self):
        self.add_noise = True
        self.mean = 0
        self.std = 1
        self.data_folder = f"data/{self.dataset}/{self.split}_{f'Noisy({self.mean}-{self.std})' if self.add_noise else ''}"


    # Model Options
    def CNN(self):
        self.kernel_size = 32
        self.channels = (16, 16, 16, 16)
    
    def SmoothCNN(self):
        self.kernel_size = 32
        self.channels = (16, 16, 16, 16)
        self.pct = 0.1 
        self.model_folder = f"models/{self.dataset}/{self.split}/{self.model}" + f"_{self.tag}" f"_pct{self.pct}"

    def save_options(self, folder):
        os.makedirs(f'{folder}', exist_ok=True)
        with open(f'{folder}/options.json', 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def load_options(self, folder):
        with open(f'{folder}/options.json', 'r') as f:
            self.__dict__.update(json.load(f))

if __name__ == "__main__":
    opt = Options(dataset='ECG', model='SmoothCNN', tag="Something")
    for k, v in opt.__dict__.items():
        print(k, v)
