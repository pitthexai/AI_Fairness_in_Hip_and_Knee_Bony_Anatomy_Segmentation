import cv2
import pydicom
import nibabel
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class BonyAnatomyJointSegmentationDataset(Dataset):
    def __init__(self, root_dir, ids, transforms=None):
        self.root_dir = root_dir
        self.pids = ids
        self.transforms = transforms

    def load_dicom(self, path):
        dicom_img = pydicom.dcmread(path)
        return dicom_img.pixel_array.astype(np.float32)

    def load_nii(self, path):
        nii_annot = nibabel.load(path)
        nii_annot_data = nii_annot.get_fdata()

        if len(nii_annot_data.shape) == 3 and nii_annot_data.shape[-1] > 1:
            if nii_annot_data.shape[-1] == 2:
                nii_annot_data = nii_annot_data[:, :, 1]
            else:
                nii_annot_data = nii_annot_data[:, :, nii_annot_data.shape[-1]//2]

            nii_annot_data = np.expand_dims(nii_annot_data, axis=-1)


        nii_annot_data = cv2.rotate(nii_annot_data, cv2.ROTATE_90_CLOCKWISE)
        nii_annot_data = cv2.flip(nii_annot_data, 1)
        return nii_annot_data

    def get_file_path(self, filename):
        return os.path.join(self.root_dir, filename)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):

        image = self.load_dicom(self.get_file_path(os.path.join("Images/", str(self.pids[idx]) + ".dcm")))
        mask = self.load_nii(self.get_file_path(os.path.join("Annotations", str(self.pids[idx]) + ".nii.gz")))

#         if len(np.unique(mask)) != self.num_classes:
#             print(self.pids[idx])
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
    

        return image.type(torch.FloatTensor), mask.long()
    
class StratifiedSampler:
    """
    Based on this Pytorch discussion board post
    https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/6
    """
    def __init__(self, stratify_on, batch_size):
        self.stratify_on = stratify_on
        self.batch_size = batch_size
        self.nsplits = int(len(stratify_on) / batch_size)

    def gen_stratified_sample(self):
        s = StratifiedKFold(n_splits = self.nsplits)

        X = np.arange(0, len(self.stratify_on))
        s.get_n_splits(X, self.stratify_on)
        for train_idx, valid_idx in s.split(X, self.stratify_on):
            yield valid_idx

    def __iter__(self):
        return iter(self.gen_stratified_sample())

    def __len__(self):
        return self.nsplits