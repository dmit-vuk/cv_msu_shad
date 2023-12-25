import os

import numpy as np
from skimage.io import imread
from tqdm import tqdm


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=mean, std=std)
])

train_transform = transforms.Compose([
    #transforms.RandomAffine(degrees=20, shear=0.05),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=mean, std=std)
])


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

class BirdsDataset(Dataset):
    def __init__(self, mode, dataset_path, gt=None, transformations=test_transform,
                 class_size=50, train_split=0.8):
        super().__init__()

        self.transformations = transformations
        self.items = []

        if mode == 'train' or mode == 'val':
            gt = np.array(list(gt.items()))
            gt_arr = np.zeros(gt.shape, dtype="object")
            for i in range(gt.shape[0]):
                gt_arr[i, :] = [os.path.join(dataset_path, gt[i, 0]), int(gt[i, 1])]
            classes = np.split(gt_arr, class_size)

            if mode == 'train':
                indecies = np.arange(int(class_size * train_split))
            else:
                indecies = np.arange(int(class_size * train_split), class_size)
            for c in classes:
                self.items += list(c[indecies])
        elif mode == 'test':
            filenames = os.listdir(dataset_path)
            for filename in filenames:
                self.items += [[os.path.join(dataset_path, filename), filename]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label_file = self.items[index]
        img = torchvision.io.read_image(img_path).type(torch.float) / 255
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transformations is not None:
            return self.transformations(img), label_file
        return img, label_file

class EfficientNett(nn.Module):
    def __init__(self, num_classes, transfer=True, freeze='most'):
        super().__init__()
        #self.model = models.efficientnet_b4(weights='IMAGENET1K_V1' if transfer else None)
        self.model = models.convnext_small(weights='IMAGENET1K_V1' if transfer else None)
        #self.model = models.efficientnet_v2_m(weights='IMAGENET1K_V1' if transfer else None)
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],# if isinstance(self.model.classifier[0], torch.nn.LayerNorm)
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.LazyLinear(512),
            nn.GELU(),
            #nn.BatchNorm1d(512),

            # nn.LazyLinear(256),
            # nn.ReLU(),
            # nn.BatchNorm1d(256),

            #nn.Dropout(p=0.1),
            nn.LazyLinear(num_classes)
        )

        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = True

        if freeze == 'last':
            for child in list(self.model.children())[0][:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most_2':
            for child in list(self.model.children())[0][:-2]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most':
            for child in list(self.model.children())[0][:-5]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class TrainModule(pl.LightningModule):
    def __init__(self, freeze="most", fast_train=False):
        super(TrainModule, self).__init__()
        self.model = EfficientNett(50, transfer=not fast_train, freeze=freeze)
        self.loss = nn.CrossEntropyLoss()

        self.training_step_loss = []
        self.training_step_acc = []
        self.training_step_lens = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_lens = []

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        logs = {
            'train_loss': loss.detach(),
            'train_acc': acc.detach()
        }
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)

        self.training_step_acc.append((logits.argmax(dim=1) == y).sum().reshape(-1))
        self.training_step_lens.append(torch.tensor([y.shape[0]]))
        self.training_step_loss.append(loss.reshape(-1))
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)

        self.validation_step_acc.append((logits.argmax(dim=1) == y).sum().reshape(-1))
        self.validation_step_lens.append(torch.tensor([y.shape[0]]))
        self.validation_step_loss.append(loss.reshape(-1))
        return {'val_loss': loss, 'val_acc': acc}

    def on_training_epoch_end(self):
        loss = torch.cat(self.training_step_loss).mean()
        acc = torch.cat(self.training_step_acc).sum() / torch.cat(self.training_step_lens).sum()
        print(f"Train Loss: {loss}")
        print(f"Train Accuracy: {acc}")
        self.training_step_loss.clear()
        self.training_step_acc.clear()
        self.training_step_lens.clear()

        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', acc, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        loss = torch.cat(self.validation_step_loss).mean()
        acc = torch.cat(self.validation_step_acc).sum() / torch.cat(self.validation_step_lens).sum()
        print(f"Val Loss: {loss}")
        print(f"Val Accuracy: {acc}")

        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_acc', acc, on_epoch=True, on_step=False)

        self.validation_step_loss.clear()
        self.validation_step_acc.clear()
        self.validation_step_lens.clear()

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)#, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=5,
            verbose=True,
            threshold= 1e-2
        )
        lr_dict = {
            # The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]

def train_classifier(train_gt, train_img_dir, fast_train=False):
    """ Callbacks and Trainer """

    checkpoint_callback = ModelCheckpoint(dirpath='./',
                                          filename='{epoch}-{val_acc:.3f}',
                                          monitor='val_acc', mode='max', save_top_k=1, verbose=True)

    max_epochs = 1 if fast_train else 60
    callbacks = [] if fast_train else [checkpoint_callback]
    trainer = pl.Trainer(
        accelerator="gpu" if not fast_train else 'cpu',
        devices = 1,
        callbacks=callbacks,
        max_epochs=max_epochs,
        logger=False, 
        enable_checkpointing=False,
    )

    train_dataset = BirdsDataset("train", train_img_dir, train_gt, transformations=train_transform)
    val_dataset = BirdsDataset("val", train_img_dir, train_gt, transformations=test_transform)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = TrainModule(fast_train=fast_train, freeze='last')
    trainer.fit(model, train_dataloader, val_dataloader)
    return model

def predict(model, dataloader):
    preds = {}
    for i, batch in enumerate(tqdm(dataloader)):
        images, files = batch
        with torch.no_grad():
            output = model(images)
            pred = output.argmax(dim=1).cpu().numpy()
            for k in range(len(pred)):
                preds[files[k]] = np.array([pred[k]])
    return preds

def classify(model_path, test_img_dir):
    model = EfficientNett(50, transfer=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    ds_test = BirdsDataset('test', dataset_path=test_img_dir)
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False)
    res = predict(model, dl_test)
    return res

if __name__ == '__main__':
    train_img_dir = 'tests/00_test_img_input/train/images'
    model_filename = 'birds_model.ckpt'
    res = classify(model_filename, train_img_dir)