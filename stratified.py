import argparse

from config import TrainingConfig
from bonyanatomy.pipeline import TrainingPipeline
from bonyanatomy.dataset import StratifiedSampler
from bonyanatomy.labels import *

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Encoder backbone for U-Net")
    parser.add_argument("data_root", help="CSV file containing demographic information for training")
    parser.add_argument("img_root", help="Root of the imaging data and annotations")
    parser.add_argument("outdir", help="Directory to save training, validation, and testing splits for experimentations")
    return parser

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    knee_config = TrainingConfig()

    # Set configuration for Knee experiments
    # SEX
    knee_config.labels = KneeAnatomy()
    knee_config.encoder_backbone = args.model
    knee_config.train_set = f"{args.data_root}/knee/knee_train_all.csv"
    knee_config.valid_set =  f"{args.data_root}/knee/knee_valid_all.csv"
    knee_config.test_set = f"{args.data_root}/knee/knee_test_all.csv"
    knee_config.imaging_root = f"{args.img_root}/Knee/"
    knee_config.outdir = args.outdir


    knee_baseline = TrainingPipeline(f"KneeStratified_Gender_{knee_config.encoder_backbone}", knee_config, StratifiedSampler, "P02SEX")
    knee_baseline.run()
    knee_baseline.save()

      # RACE
    knee_config.labels = KneeAnatomy()
    knee_config.encoder_backbone = args.model
    knee_config.train_set = f"{args.data_root}/knee/knee_train_all.csv"
    knee_config.valid_set =  f"{args.data_root}/knee/knee_valid_all.csv"
    knee_config.test_set = f"{args.data_root}/knee/knee_test_all.csv"
    knee_config.imaging_root = f"{args.img_root}/Knee/"
    knee_config.outdir = args.outdir


    knee_baseline = TrainingPipeline(f"KneeStratified_Race_{knee_config.encoder_backbone}", knee_config, StratifiedSampler, "P02RACE")
    knee_baseline.run()
    knee_baseline.save()

    # AGE
    knee_config.labels = KneeAnatomy()
    knee_config.encoder_backbone = args.model
    knee_config.train_set = f"{args.data_root}/knee/knee_train_all.csv"
    knee_config.valid_set =  f"{args.data_root}/knee/knee_valid_all.csv"
    knee_config.test_set = f"{args.data_root}/knee/knee_test_all.csv"
    knee_config.imaging_root = f"{args.img_root}/Knee/"
    knee_config.outdir = args.outdir


    knee_baseline = TrainingPipeline(f"KneeStratified_Age_{knee_config.encoder_backbone}", knee_config, StratifiedSampler, "V00AGE_GROUP")
    knee_baseline.run()
    knee_baseline.save()

    hip_config = TrainingConfig()

    # Set configuration for Hip experiments
    # SEX
    hip_config.labels = HipAnatomy()
    hip_config.encoder_backbone = args.model
    hip_config.train_set = f"{args.data_root}/hip/hip_train_all.csv"
    hip_config.valid_set =  f"{args.data_root}/hip/hip_valid_all.csv"
    hip_config.test_set = f"{args.data_root}/hip/hip_test_all.csv"
    hip_config.imaging_root = f"{args.img_root}/Hip/"
    hip_config.outdir = args.outdir

    hip_baseline = TrainingPipeline(f"HipStratified_Gender_{hip_config.encoder_backbone}", hip_config, StratifiedSampler, "P02SEX")
    hip_baseline.run()
    hip_baseline.save()

    # RACE 
    hip_config.labels = HipAnatomy()
    hip_config.encoder_backbone = args.model
    hip_config.train_set = f"{args.data_root}/hip/hip_train_all.csv"
    hip_config.valid_set =  f"{args.data_root}/hip/hip_valid_all.csv"
    hip_config.test_set = f"{args.data_root}/hip/hip_test_all.csv"
    hip_config.imaging_root = f"{args.img_root}/Hip/"
    hip_config.outdir = args.outdir

    
    hip_baseline = TrainingPipeline(f"HipStratified_Race_{hip_config.encoder_backbone}", hip_config, StratifiedSampler, "P02RACE")
    hip_baseline.run()
    hip_baseline.save()


    # AGE
    hip_config.labels = HipAnatomy()
    hip_config.encoder_backbone = args.model
    hip_config.train_set = f"{args.data_root}/hip/hip_train_all.csv"
    hip_config.valid_set =  f"{args.data_root}/hip/hip_valid_all.csv"
    hip_config.test_set = f"{args.data_root}/hip/hip_test_all.csv"
    hip_config.imaging_root = f"{args.img_root}/Hip/"
    hip_config.outdir = args.outdir

    
    hip_baseline = TrainingPipeline(f"HipStratified_Age_{hip_config.encoder_backbone}", hip_config, StratifiedSampler, "V00AGE_GROUP")
    hip_baseline.run()
    hip_baseline.save()


