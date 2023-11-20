# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import csv
import json
import tqdm
import pickle
import typing

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

from skimage.io import imread
from skimage.transform import resize
from PIL import Image


CLASSES_CNT = 205


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.samples = []
        for folder in root_folders:
            types = sorted(os.listdir(folder))
            for type in types:
                folder_type = os.path.join(folder, type)
                for name in sorted(os.listdir(folder_type)):
                    self.samples.append((os.path.join(folder_type, name), self.class_to_idx[type]))
        ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = {}
        for i in range(len(self.classes)):
            self.classes_to_samples[i] = []
        for i, class_idx in(enumerate(self.samples)):
            self.classes_to_samples[class_idx[1]].append(i)
        
        ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = A.Compose([#A.augmentations.transforms.MotionBlur(p=0.25),
                                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.7),
                                    #A.Blur(2),
                                    A.augmentations.geometric.rotate.Rotate(20,p=0.5), 
                                    A.augmentations.Resize(130, 130), 
                                    A.augmentations.crops.transforms.RandomCrop(128,128),
                                    A.Normalize(mean=mean, std=std),
                                    ToTensorV2(),])

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        path, class_idx = self.samples[index]
        image = imread(path).astype(np.float32) / 255
        image = self.transform(image=image)["image"]
        return image, path, class_idx
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        f = open(path_to_classes_json)
        classes_json = json.load(f)
        class_to_idx = {}
        classes = []
        for i, key in enumerate(classes_json.keys()):
            class_to_idx[key] = i
            classes.append(key)
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        ### YOUR CODE HERE - список путей до картинок
        self.samples = []
        for name in sorted(os.listdir(root)):
            self.samples.append(name)
        
        ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = A.Compose([A.augmentations.Resize(128, 128), 
                                    A.Normalize(mean=mean, std=std),
                                    ToTensorV2(),])
        self.targets = None
        _, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        if annotations_file is not None:
            ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            self.targets = {}
            annotations = pd.read_csv(annotations_file)
            for path_to_img in self.samples:
                name = path_to_img.split('/')[-1]
                self.targets[path_to_img] = class_to_idx[annotations[annotations['filename'] == name].iloc[0, 1]]

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        class_name = self.samples[index]
        path = os.path.join(self.root, class_name)
        image = imread(path).astype(np.float32) / 255
        image = self.transform(image=image)["image"]
        class_idx = -1
        if self.targets is not None:
            class_idx = self.targets[class_name]
        return image, class_name, class_idx
    
    def __len__(self):
        return len(self.samples)

def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)

class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion=None, internal_features=1024, freeze='last'):
        super(CustomNetwork, self).__init__()
        ### YOUR CODE HERE
        self.loss = nn.CrossEntropyLoss()
        self.features_criterion = features_criterion
        
        self.eff_net = models.efficientnet_b4(weights=None)
        #eff_net = models.efficientnet_b4(weights=None)
        if freeze == 'last':
            for child in list(self.eff_net.children())[0][:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most':
            for child in list(self.eff_net.children())[0][:-3]:
                for param in child.parameters():
                    param.requires_grad = False
        
        self.eff_net.head = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.LazyLinear(internal_features),)
        
        self.eff_net.classifier = nn.Sequential(        
                nn.ReLU(),
                nn.LazyLinear(205),
        )

        self.training_step_loss = []
        self.training_step_acc = []
        self.training_step_lens = []
        self.training_step_poss = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_lens = []
        self.validation_step_poss = []

    def forward(self, x):
        x = self.eff_net.features(x)
        out = self.eff_net.head(x)
        if self.features_criterion is not None:
            return self.eff_net.classifier(out), out
        return self.eff_net.classifier(out), None

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        ### YOUR CODE HERE
        logits, _ = self.forward(x)
        y_pred = logits.argmax(axis=1).detach()
        return y_pred.numpy()
    
    def training_step(self, train_batch, batch_idx):
        x, _,  y = train_batch
        logits, embeds = self.forward(x)
        if self.features_criterion is not None:
            out1 = embeds
            y1 = y
            idx = np.arange(y.shape[0])
            np.random.shuffle(idx)
            out2 = embeds[idx]
            y2 = y[idx] 
            loss = self.loss(logits, y) + self.features_criterion((out1, out2), (y1, y2))
        else:
            loss = self.loss(logits, y)

        y_pred = logits.argmax(axis=1)
        self.training_step_acc.append((y_pred == y).sum().reshape(-1))
        self.training_step_lens.append(torch.tensor([y.shape[0]]))
        self.training_step_loss.append(loss.reshape(-1))
        if self.features_criterion is not None:
            self.training_step_poss.append((y1 == y2).sum().reshape(-1))
        return {'loss' : loss, 'y_pred' : y_pred.detach(), 'target' : y}
    
    def on_train_epoch_end(self):
        loss = torch.cat(self.training_step_loss).mean()
        acc = torch.cat(self.training_step_acc).sum() / torch.cat(self.training_step_lens).sum()
        print(f"Train Loss: {loss}")
        print(f"Train Accuracy: {acc}")
        self.training_step_loss.clear()
        self.training_step_acc.clear()
        self.training_step_lens.clear()
        if self.features_criterion is not None:
            pos = torch.cat(self.training_step_poss).sum() / torch.cat(self.training_step_lens).sum()
            print(f"Train Posirives: {pos}")
            self.training_step_poss.clear()

    def validation_step(self, val_batch, batch_idx):
        x, _, y = val_batch
        logits, embeds = self.forward(x)
        if self.features_criterion is not None:
            out1 = embeds
            y1 = y
            idx = np.arange(y.shape[0])
            np.random.shuffle(idx)
            out2 = embeds[idx]
            y2 = y[idx] 
            loss = self.loss(logits, y) + self.features_criterion((out1, out2), (y1, y2))
        else:
            loss = self.loss(logits, y)
        y_pred = logits.argmax(axis=1).detach()
        acc = calc_metric(y, y_pred, 'all', None)

        self.validation_step_acc.append((y_pred == y).sum().reshape(-1))
        self.validation_step_lens.append(torch.tensor([y.shape[0]]))
        self.validation_step_loss.append(loss.reshape(-1))
        
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        loss = torch.cat(self.validation_step_loss).mean()
        print(torch.cat(self.validation_step_lens).sum())
        acc = torch.cat(self.validation_step_acc).sum() / torch.cat(self.validation_step_lens).sum()
        print(f"Val Loss: {loss}")
        print(f"Val Accuracy: {acc}")
        self.validation_step_loss.clear()
        self.validation_step_acc.clear()
        self.validation_step_lens.clear()

    def test_step(self, val_batch, batch_idx):
        x, _, y = val_batch
        logits, _ = self.forward(x)
        loss = self.loss(logits, y)
        y_pred = logits.argmax(axis=1).detach()
        acc = calc_metric(y_pred, y, 'all', None)
        return {"test_loss": loss, "test_acc": acc}
    

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold= 1e-2)
        lr_dict = {"scheduler": lr_scheduler,  "interval": "epoch", "frequency": 1, "monitor": "val_loss"}
        return [optimizer], [lr_dict]


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    ### YOUR CODE HERE
    model = CustomNetwork(freeze='most')
    training_data = DatasetRTSD(['cropped-train'], 'classes.json')
    
    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath="runs/pl_classifier",
        filename="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
            max_epochs=20,
            accelerator="gpu",
            devices = 1,
            #checkpoint_callback=False,
            #logger = False,
            callbacks=[MyTrainingModuleCheckpoint],
        )
    print(len(training_data))
    train_size = int(0.999 * len(training_data))
    test_size = len(training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(training_data, [train_size, test_size])
    print(len(train_dataset), len(val_dataset))
    train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val = torch.utils.data.DataLoader(val_dataset, batch_size=512, num_workers=1)
    trainer.fit(model, train, val)
    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    ### YOUR CODE HERE, results - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    test_dataset = TestData(test_folder, path_to_classes_json)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    results = []
    classes, _ = DatasetRTSD.get_classes(path_to_classes_json)
    for elem in test:
        with torch.no_grad():
            if isinstance(model, ModelWithHead):
                y_pred = np.argmax(model.model(elem[0])[0], axis=1)
            else:
                y_pred = np.argmax(model(elem[0])[0], axis=1)
        for img, path, index, pred in zip(*elem, y_pred):
            d = {"filename": path, "class": classes[pred]}
            results.append(d)
    return results

def test_classifier(model, test_folder, path_to_classes_json, annotations_file=None):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    results = apply_classifier(model, test_folder, path_to_classes_json)
    _, class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
    y_pred = np.array([])
    y_true = np.array([])
    if annotations_file is not None:
        ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
        annotations = pd.read_csv(annotations_file)
        for key, val in results.items():
            y_pred = np.append(y_pred, val)
            y_true = np.append(y_true, class_to_idx[annotations[annotations['filename'] == key].iloc[0, 1]])
    else:
        for key, val in results.items():
            y_pred = np.append(y_pred, val)
            y_true = np.append(y_true, 1)

    with open(path_to_classes_json, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)
    return total_acc, rare_recall, freq_recall

class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        self.backgrounds = os.listdir(background_path)
        self.background_path = background_path
        self.transforms = A.Compose([A.augmentations.geometric.rotate.Rotate([-15, 15],p=1,border_mode=0),
                                     A.augmentations.transforms.ColorJitter(brightness=0.87, contrast=0.4, saturation=0.87, hue=0.07,p=1),
                                    A.MotionBlur(p=0.9),
                                    A.GaussianBlur((1,5),p=0.9)])

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        r_size = np.random.randint(16,128)
        icon = resize(icon, (r_size,r_size))
        pad_size = np.random.uniform(0, 15)
        pad_size = int(r_size*0.01* pad_size)
        icon = np.pad(icon, pad_size)[:, :, pad_size : pad_size+4]
        icon, mask = icon[:, :, 0:3], icon[:, :, 3:4]
        transformed = self.transforms(image=icon, mask=mask)
        icon = transformed['image']
        mask = transformed['mask']
        mask = np.dstack((mask, mask, mask))
        bg = imread(os.path.join(self.background_path, np.random.choice(self.backgrounds))).astype(np.float32) / 255
        crop = A.augmentations.crops.transforms.RandomCrop(int(r_size*1.45),int(r_size*1.45))
        bg = crop(image=bg)['image']
        w, h, c = bg.shape
        pos1 = np.random.randint(0, w - mask.shape[0])
        pos2 = np.random.randint(0, h - mask.shape[1])
        full_mask = np.zeros((w, h, 3))
        full_mask[pos1:pos1+mask.shape[0], pos2:pos2+mask.shape[1], :] = mask
        full_icon = np.zeros((w,h,3))
        full_icon[pos1:pos1+mask.shape[0], pos2:pos2+mask.shape[1], :] = icon
        result = full_mask * full_icon + (1-full_mask)*bg
        return result


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE
    icon_path, out_path, back_path, num_samples = args[:]
    icon = imread(icon_path).astype(np.float32) / 255
    if icon.shape[-1] != 4:
        icon_new = np.ones((icon.shape[0], icon.shape[1], 4)).astype(np.float32)
        icon_new[..., 0] = icon_new[..., 1] = icon_new[..., 2] = icon[..., 0]
        icon_new[..., 3] = icon[..., 1]
        icon = icon_new
    generator = SignGenerator(back_path)
    lst = os.listdir(out_path)
    for i in range(num_samples):
        img = np.clip(generator.get_sample(icon)*255,0,255).astype('uint8')
        im = Image.fromarray(img)
        fold = icon_path.split('/')[-1].split('.png')[0]
        new_path = os.path.join(out_path, fold)
        #print(out_path, fold, new_path)
        if fold not in lst:
            os.mkdir(new_path)
            lst = os.listdir(out_path)
        im.save(os.path.join(new_path, str(i) + ".png"))


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(32) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    model = CustomNetwork(freeze='most')
    training_data_real = DatasetRTSD(['cropped-train'], 'classes.json')
    training_data_synt = DatasetRTSD(['synt'], 'classes.json')
    print(len(training_data_real), len(training_data_synt))
    training_data = torch.utils.data.ConcatDataset([training_data_real, training_data_synt])

    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath="runs/pl_classifier_synt",
        filename="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices = 1,
            #checkpoint_callback=False,
            #logger = False,
            callbacks=[MyTrainingModuleCheckpoint],
        )

    print(len(training_data))
    train_size = int(0.999 * len(training_data))
    test_size = len(training_data) - train_size
    train_set, val_set = torch.utils.data.random_split(training_data, [train_size, test_size])
    
    
    train = torch.utils.data.DataLoader(train_set, batch_size=512, num_workers=16, shuffle=True)
    val = torch.utils.data.DataLoader(val_set, num_workers=16, batch_size=512)
    trainer.fit(model, train, val)
    return model

class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs, labels):
        ### YOUR CODE HERE
        output1, output2 = outputs
        label1, label2 = labels

        distances_pos = (output2 - output1).square().sum(axis=-1)
        distances_neg = F.relu(self.margin - (distances_pos + self.eps).sqrt()).square()
        loss = 0.5 * torch.where(label1 == label2, distances_pos, distances_neg)
        return loss.mean()


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        self.dataset = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.all_classes = np.arange(len(data_source.classes_to_samples))
        #print('LEN ', len(self.all_classes))
    
    def __iter__(self):
        ### YOUR CODE HERE
        np.random.shuffle(self.all_classes)
        sample_classes = self.all_classes[:self.classes_per_batch]
        batch = []
        for class_id in sample_classes:
            cnt = len(self.dataset.classes_to_samples[class_id])
            if  cnt > self.elems_per_class:
                samples = np.array(self.dataset.classes_to_samples[class_id])
                np.random.shuffle(samples)
                batch += list(samples[:self.elems_per_class])
            else:
                num = self.elems_per_class
                while num > cnt:
                    batch += self.dataset.classes_to_samples[class_id]
                    num -= cnt
                batch += self.dataset.classes_to_samples[class_id][:num]
        yield batch

def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE
    features_criterion = FeaturesLoss(margin=1)
    model = CustomNetwork(features_criterion=features_criterion, freeze='most')
    training_data_real = DatasetRTSD(['cropped-train'], 'classes.json')
    training_data_synt = DatasetRTSD(['synt'], 'classes.json')
    print(len(training_data_real), len(training_data_synt))
    training_data = torch.utils.data.ConcatDataset([training_data_real, training_data_synt])

    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath="runs/pl_classifier_better",
        filename="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices = 1,
            #checkpoint_callback=False,
            #logger = False,
            callbacks=[MyTrainingModuleCheckpoint],
        )
    
    print(len(training_data))
    train_size = int(0.999 * len(training_data))
    test_size = len(training_data) - train_size
    train_set, val_set = torch.utils.data.random_split(training_data, [train_size, test_size])
    
    sampler = CustomBatchSampler(train_set, elems_per_class=30, classes_per_batch=10)
    train = torch.utils.data.DataLoader(train_set, batch_size=300, num_workers=16, shuffle=True, sampler=sampler)
    val = torch.utils.data.DataLoader(val_set, num_workers=16, batch_size=512)
    trainer.fit(model, train, val)
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        features_criterion = FeaturesLoss(margin=1)
        self.model = CustomNetwork(features_criterion=features_criterion, freeze='most')
        self.model.load_state_dict(torch.load(nn_weights_path, map_location='cpu'))

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        #self.knn = pickle.load(open('knn_model.bin', 'rb'))

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        model_pred, features = self.model(imgs) ### YOUR CODE HERE - предсказание нейросетевой модели
        features = features / np.linalg.norm(features.detach().numpy(), axis=1)[:, None]
        knn_pred = self.knn.predict(features) ### YOUR CODE HERE - предсказание kNN на features
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        self.dataset = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        ### YOUR CODE HERE
        batch = []
        for _, images in self.dataset.classes_to_samples.items():
            cnt = len(images)
            if  cnt > self.examples_per_class:
                samples = np.array(images)
                np.random.shuffle(samples)
                batch += list(samples[:self.examples_per_class])
            else:
                num = self.examples_per_class
                while num > cnt:
                    batch += images
                    num -= cnt
                batch += images[:num]
        for elem in batch:
            yield elem


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE

if __name__ == '__main__':
    # model = train_simple_classifier()
    # torch.save(model, 'model.ckpt')
    # model = torch.load("model.ckpt")
    # torch.save(model.state_dict(), "./simple_model.pth") 

    # model = train_synt_classifier()
    # torch.save(model, 'model_synt.ckpt')
    # model = torch.load("model_synt.ckpt")
    # torch.save(model.state_dict(), "./simple_model_with_synt.pth")

    # model = train_better_model()
    # torch.save(model, 'model_better.ckpt')
    # model = torch.load("model_better.ckpt")
    # torch.save(model.state_dict(), "./improved_features_model.pth")

    generate_all_data('synt', 'icons', 'background_images')
    training_data = DatasetRTSD(['synt'], 'classes.json')
    print(len(training_data))