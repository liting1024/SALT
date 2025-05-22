import time
import os 
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, params, log_path, tensorboard=False):
        self.tensorboard = tensorboard
        config_name = params.config_yaml.split('/')[-1].split('.')[0]
        base = f"{log_path}/logs/{params.data_name}"
        if tensorboard:
            self.log_dir = f"{base}/{config_name}"
            self.writer = SummaryWriter(self.log_dir)
        self.save_dir = f'{log_path}/pt/{params.data_name}/{config_name}'
        os.makedirs(self.save_dir, exist_ok='True')
        
    def save_model(self, model, optimizer, epoch, save_name):
        save_path = f'{self.save_dir}/{epoch}_{save_name}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f'Model saved to {save_path}')

    def load_model(self, model, optimizer, load_name):
        load_path = f'{self.save_dir}/{load_name}.pth'
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Model loaded from {load_path}')
        return model, optimizer, epoch
    
    def write(self, metrics, mode, epoch):
        if self.tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{key}/{mode}', value, epoch)
            self.writer.flush()
        else:
            metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
            end = '\n' if mode == 'Test' else ''
            self.write_txt(f'{mode} epoch: {epoch}, {metrics_str}{end}')