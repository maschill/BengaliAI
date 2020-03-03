from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import pandas as pd 

from PIL import Image
from torch import LongTensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision import transforms 

DS_MEAN = 226.83368
DS_STD = 59.658222

DATA_ROOT = "/app/data/graphemes/"
csv_file = "/app/data/train.csv"

class GraphemesDataset(Dataset):
    """
    Base class for loading images

    Parameters
    ----------
    train_df: pandas.DataFrame
        A dataframe containing image names and labels
    root_dir: str
        A string containing the location of the images
    transform: 
        pytorch.torchvision.transforms applied to the images
    """ 
    def __init__(self, train_df, root_dir, transform): 
        self.train_df = train_df
        self.transform = transform
        self.mean = DS_MEAN
        self.std = DS_STD
        self.grapheme_labels = train_df["grapheme_root"].values
        self.vowel_labels = train_df["vowel_diacritic"].values
        self.consonant_labels = train_df["consonant_diacritic"].values

    def _load_img(self,idx): 
        return None

    def __len__(self): 
        return(len(self.train_df))

    def __getitem__(self, idx): 
        image = self._load_img(idx)

        img = self.transform(image)
        img = (img-self.mean)/self.std

        labels = LongTensor([
            self.grapheme_labels[idx], 
            self.vowel_labels[idx], 
            self.consonant_labels[idx],
            ])

        return img, labels

class PreloadedGraphemes(GraphemesDataset): 
    """
    Class for loading images from 128x128 numpy.memmap 

    Parameters
    ----------
    train_df: pandas.DataFrame
        A dataframe containing image names and labels
    root_dir: str
        A string containing the location of the memmap
    transform: 
        pytorch.torchvision.transforms applied to the images
    """
    def __init__(self, train_df, root_dir, transform): 
        super().__init__(train_df, root_dir, transform)
        self.images = np.memmap(
            f"{root_dir}128_128_cropped.npy",
            mode="r",
            dtype=np.float32,
            shape=(200840, 128, 128) 
        )

    def __len__(self): 
        return(len(self.train_df))

    def _load_img(self, idx): 
        img_idx = int(self.train_df.iloc[idx, 0].split("_")[1])
        image = Image.fromarray((self.images[img_idx].astype(np.float32)))
        return image

class ImageGraphemes(GraphemesDataset):
    """
    Class for loading images from image files on disk

    Parameters
    ----------
    train_df: pandas.DataFrame
        A dataframe containing image names and labels
    root_dir: str
        A string containing the location of the images
    transform: 
        pytorch.torchvision.transforms applied to the images
    """
    def __init__(self, train_df, root_dir, transform): 
        super().__init__(train_df, root_dir, transform)
        self.root_dir = root_dir

    def _load_img(self, idx): 
        img_name = self.train_df.iloc[idx, 0]
        image = Image.open(f"{self.root_dir}{img_name}.png")

class BengaliGraphemes(Object):
    """
    Class combining dataloaders

    Parameters
    ----------
    batch_size: int
    csv_file: str
        Filename of the csv containing image names and labels
    img_size: int
        size the images will be resized to
    root_dir: str
        Root directory containing the images
    preloaded: bool
        wether to use preloaded images on memmap or load from image files
    """
    def __init__(self, batch_size, csv_file=csv_file, img_size=128, root_dir=DATA_ROOT, preloaded=True):
        df = pd.read_csv(csv_file)

        print(df.columns)
        df["grapheme_root"] = pd.to_numeric(df["grapheme_root"])
        df["vowel_diacritic"] = pd.to_numeric(df["vowel_diacritic"])
        df["consonant_diacritic"] = pd.to_numeric(df["consonant_diacritic"])
        self.num_classes = (df["grapheme_root"].max()+1, 
                            df["vowel_diacritic"].max()+1, 
                            df["consonant_diacritic"].max()+1)

        print(len(df))
        np.random.seed(123456)
        mask = np.random.rand(len(df)) < 0.8
        
        train_df = df[mask]
        val_df = df[~mask]

        # relative frequencies:
        self.rel_frequencies = {
            "train": {
                "grapheme": (train_df["grapheme_root"].value_counts()).to_dict(),
                "vowel": (train_df["vowel_diacritic"].value_counts()).to_dict(),
                "cons": (train_df["consonant_diacritic"].value_counts()).to_dict()
            },
            "val": {
                "grapheme": (val_df["grapheme_root"].value_counts()).to_dict(),
                "vowel": (val_df["vowel_diacritic"].value_counts()).to_dict(),
                "cons": (val_df["consonant_diacritic"].value_counts()).to_dict()
            }
        }
        ds_class = None
        if preloaded: 
            ds_class=PreloadedGraphemes
        else:
            ds_class=ImageGraphemes

        train_ds = ds_class(
            train_df,
            root_dir=root_dir,
            transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomCrop(img_size, padding=4),
                transforms.Resize(img_size),
                transforms.ToTensor()
            ]) 
        )

        test_ds = ds_class(
            val_df,
            root_dir=root_dir, 
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4 
        )

        self.val_loader = DataLoader(
            test_ds,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4 
        )

        self.sizes = {
            "train": len(train_ds),
            "val": len(test_ds),
            "test": len(test_ds)
        }

        self.weights = {
            "grapheme": [dl.rel_frequencies["train"]["grapheme"][i]/dl.sizes["train"] for i in range(168)]
            "vowel": [dl.rel_frequencies["train"]["vowel"][i]/dl.sizes["train"] for i in range(11)]
            "cons": [dl.rel_frequencies["train"]["cons"][i]/dl.sizes["train"] for i in range(7)]
        }
    
    def __getitem__(self, key): 
        if key=="train": 
            return self.train_loader
        elif key=="val" or key=="test": 
            return self.val_loader

        return None

