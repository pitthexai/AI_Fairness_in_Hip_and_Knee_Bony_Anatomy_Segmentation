import os
import pandas as pd
import numpy as np
import copy
import json

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from segmentation_models_pytorch import utils as smp_utils

class TrainingPipeline:
    def __init__(self, name,  training_config, sampler=None, stratify_on=None):
        self.name = name
        self.training_config = training_config
        self.stratify_on = stratify_on
        self.sampler = sampler
        self.save_dir = f"{self.training_config.outdir}/{self.name}/"
        
        self.load_datasources()
        os.makedirs(self.save_dir, exist_ok=True)

    def load_datasources(self):
        train_data = pd.read_csv(self.training_config.train_set)
        valid_data = pd.read_csv(self.training_config.valid_set)
        test_data = pd.read_csv(self.training_config.test_set)

        self.train_split = self.training_config.dataset(self.training_config.imaging_root, train_data.id, self.training_config.transforms)
        self.valid_split = self.training_config.dataset(self.training_config.imaging_root, valid_data.id, self.training_config.transforms)
        self.test_split = self.training_config.dataset(self.training_config.imaging_root, test_data.id, self.training_config.transforms)

        self.train_loader = DataLoader(self.train_split, batch_size=self.training_config.train_batch_size)

        if self.sampler:
            print("Stratified Sampler!!")
            self.train_loader = DataLoader(self.train_split, batch_sampler=self.sampler(train_data[self.stratify_on], self.training_config.train_batch_size))

        self.valid_loader = DataLoader(self.valid_split, batch_size=self.training_config.eval_batch_size, shuffle=True)


    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_classes = self.training_config.labels.get_num_classes()
        model = copy.deepcopy(smp.Unet(encoder_name=self.training_config.encoder_backbone, encoder_weights=self.training_config.encoder_weights, in_channels=self.training_config.in_channels,
                        classes=num_classes, activation=self.training_config.activation).to(device))
        model.encoder.requires_grad_ = False
        model.decoder.requires_grad_ = False

        loss = self.training_config.loss_function()
        loss.__name__=" loss"

        metrics = []
        for metric in self.training_config.metrics:
            m = metric(num_classes=num_classes, average="macro").to(device)
            m.__name__ = m.__class__.__name__
            metrics.append(m)
        # multi_jaccard = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
        # multi_jaccard.__name__ = "iou_score"
        # metrics = [multi_jaccard]

        optimizer = self.training_config.optimizer(model.parameters(), lr=self.training_config.learning_rate)

        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=True,
        )

        max_score = 0

        self.best_model = None
        for i in range(1, self.training_config.epochs + 1):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.train_loader)
            valid_logs = valid_epoch.run(self.valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['MulticlassJaccardIndex']:
                max_score = valid_logs['MulticlassJaccardIndex']
                self.best_model = copy.deepcopy(model)
                print('Model saved!')
    
    def save(self):
        torch.save(self.best_model, f"{self.save_dir}/unet_{self.training_config.encoder_backbone}.pt")


class BiasEvaluationPipeline:
    def __init__(self, data, name, config):
        self.name = name
        self.data = data
        self.config = config
        self.results = {}
        self.protected_attribute_records, self.unique_attr = self.load_protected_attribute_data()
    
    def load_protected_attribute_data(self):
        unique_att = self.data[self.config.protected_attributes].unique()
        
        return {
            protected_att: self.data[self.data[self.config.protected_attributes] == protected_att].reset_index(drop=True) for protected_att in unique_att
        }, unique_att
    
    def evaluate_attribute(self, model, attribute_data):
        attr_ds = self.config.dataset(self.config.imaging_root, attribute_data.id, self.config.transforms)
        attr_dl = DataLoader(attr_ds, batch_size=1)

        num_classes = self.config.labels.get_num_classes()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        metrics = []
        for name, metric in self.config.eval_metrics.items():
            m = metric(num_classes=num_classes, average="macro").to(device)
            m.__name__ = name
            metrics.append(m)

        scores = {name: [] for name in self.config.eval_metrics.keys()}
        for img, annot in attr_dl:
            img = img.cuda()
            out = model(img)

            for m in metrics:
                scores[m.__name__].append(m(torch.softmax(out, dim=1), annot.cuda()).item())

        return scores

            
    def evaluate_bias(self):
        bias_metrics = {name:m() for name, m in self.config.bias_metrics.items()}
        for exp, model_pth in self.config.get_models().items():
            print("Evaluating", exp)
            self.results[exp] = {}
            attr_scores = {name:[] for name in self.config.eval_metrics.keys()}
            if type(model_pth) is dict:
                for attr, m in model_pth.items():
                    model = torch.load(m)
                    records = self.protected_attribute_records[attr]
                    scores = self.evaluate_attribute(model, records)

                    for name, value in scores.items():
                        values =  np.array(value).mean()
                        self.results[exp][f"{name}_{attr}"] = values
                        attr_scores[name].append(values)

            elif type(model_pth) is str:
                model = torch.load(model_pth)
                for attribute, records in self.protected_attribute_records.items():
                    scores = self.evaluate_attribute(model, records)
                    for name, value in scores.items():
                        values =  np.array(value).mean()
                        self.results[exp][f"{name}_{attribute}"] = values
                        attr_scores[name].append(values)
            else:
                raise TypeError
            
            for name, metric in bias_metrics.items():
                for attr, values in attr_scores.items():
                    self.results[exp][f"{attr}_{name}"] = metric.compute(np.array(values))
            
    def save(self, path):
        with open(f"{path}/{self.name}.json", "w") as f:
            json.dump(self.results, f)