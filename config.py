import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Type

from bonyanatomy.dataset import BonyAnatomyJointSegmentationDataset
from bonyanatomy.bias_metrics import SkewedErrorRatio, StandardDeviation
from dataclasses import dataclass, field

import torch

import torchmetrics

@dataclass
class TrainingConfig:
    epochs = 50
    random_state = 42
    image_size = 224
    train_batch_size = 16
    eval_batch_size = 16
    learning_rate = 5e-04
    optimizer = torch.optim.Adam
    loss_function = torch.nn.CrossEntropyLoss
    dataset = BonyAnatomyJointSegmentationDataset
    activation = None
    in_channels = 1
    encoder_weights = "imagenet"
    metrics = [torchmetrics.classification.MulticlassJaccardIndex, torchmetrics.classification.Dice]
    transforms = A.Compose([A.Resize(image_size, image_size), ToTensorV2()])

    
@dataclass
class BiasEvalBaseConfig:
    dataset = BonyAnatomyJointSegmentationDataset
    eval_metrics: Dict[str, Type] = field(default_factory=lambda: {
        "IoU": torchmetrics.classification.MulticlassJaccardIndex,
        "Dice": torchmetrics.classification.Dice
    })
    bias_metrics: Dict[str, Type] = field(default_factory=lambda: {
        "SER": SkewedErrorRatio,
        "SD": StandardDeviation
    })
    image_size = 224
    transforms = A.Compose([A.Resize(image_size, image_size), ToTensorV2()])

    def __post_init__(self):
        self.bias_metrics = {
            "SER": SkewedErrorRatio,
            "SD": StandardDeviation
        }

        self.eval_metrics =  {
            "IoU": torchmetrics.classification.MulticlassJaccardIndex,
            "Dice": torchmetrics.classification.Dice
         }

class ResNet18SexGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "P02SEX"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes

    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_resnet18/unet_resnet18.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Gender_resnet18/unet_resnet18.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Gender_resnet18/unet_resnet18.pt",
            "group": {
                "Male": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_SexGroup_Male_resnet18/unet_resnet18.pt",
                "Female": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_SexGroup_Female_resnet18/unet_resnet18.pt"
            }
        }

class EfficientNetB0SexGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "P02SEX"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes
    
    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_efficientnet-b0/unet_efficientnet-b0.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Gender_efficientnet-b0/unet_efficientnet-b0.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Gender_efficientnet-b0/unet_efficientnet-b0.pt",
            "group": {"Male": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_SexGroup_Male_efficientnet-b0/unet_efficientnet-b0.pt",
                    "Female": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_SexGroup_Female_efficientnet-b0/unet_efficientnet-b0.pt"}
        }

    protected_attributes = "P02SEX"

class ResNet18RacialGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "P02RACE"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes
    
    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_resnet18/unet_resnet18.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Race_resnet18/unet_resnet18.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Race_resnet18/unet_resnet18.pt",
            "group": {"White_Caucasian": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_RaceGroup_White_resnet18/unet_resnet18.pt",
                    "Black_AfricanAmerican": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_RaceGroup_Black_resnet18/unet_resnet18.pt"}
        }


class EfficientNetB0RacialGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "P02RACE"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes
    
    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_efficientnet-b0/unet_efficientnet-b0.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Race_efficientnet-b0/unet_efficientnet-b0.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Race_efficientnet-b0/unet_efficientnet-b0.pt",
            "group": {"White_Caucasian": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_RaceGroup_White_efficientnet-b0/unet_efficientnet-b0.pt",
                    "Black_AfricanAmerican": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_RaceGroup_Black_efficientnet-b0/unet_efficientnet-b0.pt"}
        }

class ResNet18AgeGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "V00AGE_GROUP"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes
    
    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_resnet18/unet_resnet18.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Age_resnet18/unet_resnet18.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Age_resnet18/unet_resnet18.pt",
            "group": {"Age_50_Lower": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_50_Lower_resnet18/unet_resnet18.pt",
                    "Age_51_64": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_51_64_resnet18/unet_resnet18.pt",
                    "Age_65_79": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_65_79_resnet18/unet_resnet18.pt"}

        }


class EfficientNetB0AgeGroups(BiasEvalBaseConfig):
    def __init__(self, anatomy: str, protected_attributes: str = "V00AGE_GROUP"):
        super().__init__()
        self.anatomy = anatomy
        self.protected_attributes = protected_attributes
    
    def get_models(self):
        return {
            "baseline": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_efficientnet-b0/unet_efficientnet-b0.pt",
            "balanced":  f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Balanced_Age_efficientnet-b0/unet_efficientnet-b0.pt",
            "stratified": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Stratified_Age_efficientnet-b0/unet_efficientnet-b0.pt",
            "group": {"Age_50_Lower": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_50_Lower_efficientnet-b0/unet_efficientnet-b0.pt",
                    "Age_51_64": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_51_64_efficientnet-b0/unet_efficientnet-b0.pt",
                    "Age_65_79": f"/data_vault/hexai/ScientificReports_HipKnee_Results_v2/{self.anatomy}Baseline_AgeGroup_65_79_efficientnet-b0/unet_efficientnet-b0.pt"}

        }

