import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, train_meta_path):
        self.train_meta_path = train_meta_path
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.Y = None

    def load_data(self):
        self.train_meta = pd.read_csv(self.train_meta_path, low_memory=False)
        if 'target' in self.train_meta.columns:
            self.Y = self.train_meta['target'].values
            self.train_meta = self.train_meta.drop(columns=['target'])

    def drop_columns(self):
        columns_to_drop = [
            'patient_id', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
            'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence', 'copyright_license'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in self.train_meta.columns]
        self.train_meta = self.train_meta.drop(columns=columns_to_drop)

    def fill_missing_values(self):
        self.train_meta.fillna({
            'age_approx': self.train_meta['age_approx'].median(),
            'sex': self.train_meta['sex'].mode()[0],
            'anatom_site_general': self.train_meta['anatom_site_general'].mode()[0]
        }, inplace=True)

    def encode_categorical_columns(self):
        for col in ['sex', 'anatom_site_general', 'image_type', 'tbp_tile_type', 'tbp_lv_location',
                    'tbp_lv_location_simple']:
            self.train_meta[col] = self.label_encoder.fit_transform(self.train_meta[col].astype(str))

        categorical_cols = self.train_meta.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'isic_id':
                self.train_meta[col] = self.label_encoder.fit_transform(self.train_meta[col].astype(str))

    def scale_numerical_columns(self):
        numerical_cols = self.train_meta.select_dtypes(include=['float64', 'int64']).columns
        self.train_meta[numerical_cols] = self.scaler.fit_transform(self.train_meta[numerical_cols])

    def preprocess(self):
        self.load_data()
        self.drop_columns()
        self.fill_missing_values()
        self.encode_categorical_columns()
        self.scale_numerical_columns()
        self.train_meta['isic_id'] = self.train_meta['isic_id'].astype(str)
        return self.train_meta, self.Y


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, metadata_dict, labels, ids, image_size=(224, 224)):
        super(HDF5Dataset, self).__init__()
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.metadata_dict = metadata_dict
        self.labels = labels.astype(np.float32)
        self.ids = ids
        self.image_size = image_size

    def __getitem__(self, index):
        isic_id = self.ids[index]
        image_bytes = np.array(self.h5_file[isic_id]).tobytes()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image).astype(np.float32)
        metadata = np.array(self.metadata_dict[isic_id], dtype=np.float32)
        label = self.labels[index]
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(metadata), torch.tensor(label)

    def __len__(self):
        return len(self.ids)


class DatasetSplitter:
    def __init__(self, train_meta, Y, test_size=0.1):
        self.train_meta = train_meta
        self.Y = Y
        self.test_size = test_size

    def split(self):
        idx_positive = np.where(self.Y == 1)[0]
        idx_negative = np.where(self.Y == 0)[0]

        val_size_positive = int(len(idx_positive) * self.test_size)
        val_size_negative = int(len(idx_negative) * self.test_size)

        val_indices = np.concatenate([idx_positive[:val_size_positive], idx_negative[:val_size_negative]])
        train_indices = np.concatenate([idx_positive[val_size_positive:], idx_negative[val_size_negative:]])

        train_data = self.train_meta.iloc[train_indices].copy()
        val_data = self.train_meta.iloc[val_indices].copy()
        Y_train = self.Y[train_indices]
        Y_val = self.Y[val_indices]

        return train_data, val_data, Y_train, Y_val
