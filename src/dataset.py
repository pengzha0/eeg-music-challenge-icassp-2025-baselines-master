import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize

mne.set_log_level("ERROR")

class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix=""):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.split = split
        self.label_name = "subject_id" if task == "subject_identification" else "label"
        self.ext = ext
        
        splits = json.load(open(os.path.join(split_dir, f"splits_{task}.json")))
        self.samples = splits[split]
        

        files = []
        labels = []
        labels_id = []

        for sample in self.samples:
            #path = os.path.join(self.dataset_dir, self.subdir, sample['filename_preprocessed'])
            path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
            label = sample[self.label_name]
            labels_id.append(sample['id'])
            files.append(path)
            labels.append(label)
        self.datas = []
        self.labels = []
        self.ids = []
        
        for idx in range(len(files)):
            data_file = files[idx]
            if self.ext == "npy":
                data = np.load(data_file) 
            elif self.ext == "fif":
                data = mne.io.read_raw_fif(data_file, preload=True).get_data()
            else:
                raise ValueError(f"Extension {ext} not recognized")
            
            if split=='train':
                datas = self.split_data(data)

                self.datas.append(datas)
                for _ in range(datas.shape[0]):
                    self.labels.append(labels[idx])
                    self.ids.append(labels_id[idx])
            else:
                self.datas.append(data)
                self.labels.append(labels[idx])
                self.ids.append(labels_id[idx])

        if split=='train':
            self.datas = np.concatenate(self.datas,axis=0)
        # self.labels = np.array(self.labels)


    def split_data(self,data,overlap = 128):
        split_data = []
        t_len = data.shape[1]
        for i in range(0, t_len-1280, overlap):
            split_data.append(data[:,i:i+1280])
        return np.array(split_data)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            "id": self.ids[idx],
            "eeg": self.datas[idx],
            "label": self.labels[idx] if "test" not in self.split else -1,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample    
      
def get_loaders(args):
    
    if args.task == "subject_identification":
        splits = ["train", "val_trial"]
    elif args.task == "emotion_recognition":
        splits = ["train", "val_trial", "val_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    train_transforms = T.Compose([
        # RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])
    
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type 
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
            subdir=subdir,
            split_dir=args.split_dir,
            split=split,
            ext = ext,
            task = args.task,
            transform=train_transforms if split == "train" else test_transforms
        )
        for split in splits
    }
    
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=True if split == "train" else False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return loaders, args

def get_test_loader(args):
    
    if args.task == "subject_identification":
        splits = ["test_trial"]
    elif args.task == "emotion_recognition":
        splits = ["test_trial", "test_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=test_transforms
        ) for split in splits
    }
    
    datasets_no_transform = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=None
        ) for split in splits
    }
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return datasets_no_transform, loaders, args