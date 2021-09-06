import os
import json

import numpy as np
import torch

from PIL import Image

from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize

class MemesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        print(config)
        assert (
            self._use_images
        )

        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir

    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor
    
    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        # if "meme" in sample_info['id']:
        #     id = int(sample_info['id'].split("meme")[1]) + 10000
        # else:
        #     id = int(sample_info['id'].split("covid_memes_")[1])
        
        id = int(sample_info['id'][len('meme_'):])
        current_sample.id = torch.tensor(id, dtype=torch.int)

        label_map = {0: 'fear', 1: 'anger', 2: 'joy', 3: 'sadness', 4: 'surprise', 5: 'disgust'}
        rev_map = {'fear': 0, 'anger': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'disgust': 5}

        # if sample_info['labels'][1] == 'individual':
        #     label = torch.tensor(0, dtype=torch.long)
        # elif sample_info['labels'][1] == 'organization':
        #     label = torch.tensor(1, dtype=torch.long)
        # elif  sample_info['labels'][1] == 'community':
        #     label = torch.tensor(2, dtype=torch.long)
        # else:
        #     label = torch.tensor(3, dtype=torch.long)

        key = sample_info['labels'][0]
        label = torch.tensor(rev_map[key], dtype=torch.long)

        current_sample.targets = label

        print(self.image_db[idx])
 
        
        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample

class MemesFeatureDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        print(config)
        assert (
            self._use_images
        )

        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)
        self._data_folder = self._data_dir

    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor
    
    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        # if "meme" in sample_info['id']:
        #     id = int(sample_info['id'].split("meme")[1]) + 10000
        # else:
        #     id = int(sample_info['id'].split("covid_memes_")[1])
        
        id = sample_info['id'][len('meme_'):]
        current_sample.id = torch.tensor(id, dtype=torch.int)

        label_map = {0: 'fear', 1: 'anger', 2: 'joy', 3: 'sadness', 4: 'surprise', 5: 'disgust'}

        # if sample_info['labels'][1] == 'individual':
        #     label = torch.tensor(0, dtype=torch.long)
        # elif sample_info['labels'][1] == 'organization':
        #     label = torch.tensor(1, dtype=torch.long)
        # elif  sample_info['labels'][1] == 'community':
        #     label = torch.tensor(2, dtype=torch.long)
        # else:
        #     label = torch.tensor(3, dtype=torch.long)

        key = int(sample_info['labels'][1])
        label = torch.tensor(label_map[key], dtype=torch.long)

        current_sample.targets = label
 
        
        current_sample.image = self.image_db[idx]["images"][0]

        return current_sample