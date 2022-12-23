from torch.utils.data import Dataset
from path import Path
from scipy.io import loadmat
import torch
from utils.paths import ROOT



class ROIDataset(Dataset):
    """Dataset class for the Region of Interest (ROI) dataset."""

    brain_areas = ['LHPPA', 'RHLOC', 'LHLOC', 'RHEarlyVis', 'RHRSC', 
                   'RHOPA', 'RHPPA', 'LHEarlyVis', 'LHRSC', 'LHOPA']

    def __init__(self, user = 'CSI1'):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_dir: image directory.
            annotation_file: annotation file path.
            transform: image transformer.
        """
        self.user = user
        self.img_files_folder = ROOT / Path('data/ROIs/stim_lists')
        self.img_files_path = self.img_files_folder / f'{self.user}_stim_lists.txt'  # user order of viewed images
        with open(self.img_files_path, encoding='utf-8') as f:
            self.img_files = f.read().splitlines()
        self.img_files = [img_list for img_list in self.img_files if img_list]
        self.roi_folder = ROOT / Path(f'data/ROIs/{self.user.replace("0", "")}/mat')
        self.roi_files = self.roi_folder.files('*.mat')
        self.num_features, self.brain_feature_slices = self.calc_num_features(user)
        self.images_folder = Path('data/BOLD5000_Stimuli/Scene_Stimuli/Original_Images')

    @classmethod
    def calc_num_features(cls, user):
        roi_folder = ROOT / Path(f'data/ROIs/{user}/mat')
        roi_files = roi_folder.files('*.mat')
        roi_file = next(iter(roi_files))
        roi = loadmat(roi_file)
        num_features = 0
        brain_feature_slices = []
        for brain_area in cls.brain_areas:
            brain_num_features = roi[brain_area].shape[1]
            brain_feature_slices.append(slice(num_features, num_features + brain_num_features))
            num_features += brain_num_features
        return num_features, brain_feature_slices

    def __len__(self):
        return len(self.img_files) * len(self.roi_files)

    def __getitem__(self, index):
        """Returns data roi"""
        roi_file = self.roi_files[index % len(self.roi_files)]
        file_i = index // len(self.roi_files)
        img_fname = self.img_files[file_i]
        roi_matlab = loadmat(roi_file)
        roi_vector = torch.zeros(self.num_features)
        for brain_area, brain_slice in zip(self.brain_areas, self.brain_feature_slices):
            roi_vector[brain_slice] = torch.from_numpy(roi_matlab[brain_area][file_i]).float()  
        return dict(roi=roi_vector, img_fname=img_fname)


class ROIDatasetImageFeatures(ROIDataset):


    def __getitem__(self, i):
        data = super().__getitem__(i)
        return dict(roi=data['roi'], img=img)


if __name__ == '__main__':
    dataset = ROIDataset()
    print(dataset[0])