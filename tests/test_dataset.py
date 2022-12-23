from dataset.roi import ROIDataset

def test_load_row_roi():
    dataset = ROIDataset('CSI1')
    row = dataset[0]
    assert row['roi'].shape == (dataset.num_features,)
    assert row['img_fname'] == dataset.img_files[0]