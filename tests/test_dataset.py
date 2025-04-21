import os
import pandas as pd
import pytest
import tempfile
from PIL import Image
from src.data.dataset import ISICDataset

@pytest.fixture
def setup_dataset():
    test_dir = tempfile.TemporaryDirectory()

    annotations_file = os.path.join(test_dir.name, 'annotations.csv')
    data = {'image_id': ['img1', 'img2'], 'label': [0, 1]}
    df = pd.DataFrame(data)
    df.to_csv(annotations_file, index=False)

    img_dir = test_dir.name
    for img_id in data['image_id']:
        img_path = os.path.join(img_dir, img_id + '.jpg')
        image = Image.new('RGB', (100, 100)).save(img_path)
        image.save(img_path)

    dataset = ISICDataset(annotations_file, img_dir)

    yield dataset