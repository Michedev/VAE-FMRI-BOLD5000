from dataset.roi import ROIDataset, ROIDatasetImage
import torch

def test_load_row_roi():
    dataset = ROIDataset('CSI1')
    row = dataset[0]
    assert row['roi'].shape == (dataset.num_features,)
    assert row['img_fname'] == dataset.img_files[0]


def test_load_row_image():
    dataset = ROIDatasetImage('CSI1')
    row = dataset[0]
    assert row['roi'].shape == (dataset.num_features,)
    assert row['img'].dtype == torch.float32
    assert row['img'].min() >= 0
    assert row['img'].max() <= 1
    assert row['img'].shape == (3, 224, 224)