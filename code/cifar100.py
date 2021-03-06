import argparse
import logging
import json
import os
os.system('pip install  pytorch_lightning torchmetrics torchsummary -q --no-cache-dir')
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F

print(torch.__version__)
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        TQDMProgressBar
        )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CIFAR100Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=True, progress=False)
        for param in self.model.parameters():
            param.requires_grad = True
        num_ftrs = self.model.fc.in_features
        num_classes = 100
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x_var):
        """Forward function."""
        out = self.model(x_var)
        return out
    
    
class CIFAR100Classifier(pl.LightningModule):  

    def __init__(self, model, lr=0.001):
        """Initializes the network, optimizer and scheduler."""
        super(CIFAR100Classifier, self).__init__()  

       
        self.scheduler = None
        self.optimizer = None
        self.model = model
        self.lr=lr

        self.train_accm = Accuracy()
        self.valid_accm = Accuracy()

        self.train_acc = 0.
        self.valid_acc = 0.

        self.preds = []
        self.target = []

    def training_step(self, train_batch, batch_idx):
        """Training Step
        Args:
             train_batch : training batch
             batch_idx : batch id number
        Returns:
            train accuracy
        """

        x_var, y_var = train_batch
        output = self.model(x_var)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y_var)
        self.train_accm(y_hat, y_var)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """Testing step.

        Args:
             val_batch : val batch data
             batch_idx : val batch id
        Returns:
             validation accuracy
        """

        x_var, y_var = val_batch
        output = self.model(x_var)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y_var)
        self.valid_accm(y_hat, y_var)
        return {"loss": loss}

    def configure_optimizers(self):
        """Initializes the optimizer and learning rate scheduler.

        Returns:
             output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.lr
            )
        return [self.optimizer]

    def validation_epoch_end(self, outputs):        
        self.avg_valid_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.valid_acc = (self.valid_accm.compute() * 100).item()
        self.valid_accm.reset()


    def training_epoch_end(self, outputs):
        self.avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.train_acc = (self.train_accm.compute() * 100).item()
        self.train_accm.reset()
    

    def on_train_epoch_end(self):
        print(f'Epoch: {self.current_epoch+1}, Train Acc: {self.train_acc}, Train Loss: {self.avg_train_loss},  Valid Acc: {self.valid_acc}, Valid Loss: {self.avg_valid_loss}')


def _train(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device Type: {}".format(device))

    print("Loading Cifar100 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)


    bar = TQDMProgressBar(refresh_rate=50)
    trainer_args = {
    "enable_checkpointing": True,
    "callbacks": [bar],
    "max_epochs": args.epochs,
    "num_sanity_val_steps": 0,
    "fast_dev_run": False
                    }
    trainer = Trainer(
                    **trainer_args
                    )
    model_base = CIFAR100Base()
    model = CIFAR100Classifier(model_base, args.lr)
    print('Starting Training')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)  
    print('Finished Training')
    save_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model_base.cpu().state_dict(), save_path)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  CIFAR100Base()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir", type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--num-gpus", type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())