import pandas as pd

from bonyanatomy.pipeline import BiasEvaluationPipeline
from bonyanatomy.labels import KneeAnatomy, HipAnatomy
from config import ResNet18SexGroups, EfficientNetB0SexGroups, ResNet18RacialGroups, EfficientNetB0RacialGroups, ResNet18AgeGroups, EfficientNetB0AgeGroups


data = pd.read_csv("/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/Hip_segmentation.csv")
data = data[~data.id.isin([9002116,9025994])].reset_index(drop=True)

#Sex Groups
#RESNET
print("Running Sex Eval. for ResNet18...")
bias_sex_config = ResNet18SexGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessSex_Resnet18_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2")

# EFFICIENTNET
print("Running Sex Eval. for EfficientNet-B0...")
bias_sex_config = EfficientNetB0SexGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessSex_EfficientNetB0_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2/")

#Racial Groups
#RESNET
print("Running Race Eval. for ResNet18...")

bias_sex_config = ResNet18RacialGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessRace_Resnet18_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2")

# EFFICIENTNET
print("Running Race Eval. for EfficientNet-B0...")
bias_sex_config = EfficientNetB0RacialGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessRace_EfficientNetB0_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2/")


# Age Groups
# RESNET
print("Running Age Eval. for ResNet18...")
bias_sex_config = ResNet18AgeGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessAge_Resnet18_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2")

# EFFICIENTNET
print("Running Age Eval. for EfficientNet-B0...")
bias_sex_config = EfficientNetB0AgeGroups(anatomy="Hip")
bias_sex_config.labels = HipAnatomy()
bias_sex_config.imaging_root = "/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/Hip/"
bias_sex = BiasEvaluationPipeline(data, "FairnessAge_EfficientNetB0_Hip", bias_sex_config)
bias_sex.evaluate_bias()
bias_sex.save("results_v2/")

