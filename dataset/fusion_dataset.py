import os
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

anno_file = "/home/amax/tx/FeatureNext-master/data_process/result/pro_{}_train.txt"
palm_datasets = ["IITD", "MPD", "NTU-CP", "SMPD"]

class FusionDataset(Dataset):
    """
        | PALM_DATA_DIR
        |-- IITD
        |   |-- annotations
        |   |   |-- fourk_train.json
        |   |   |-- fourk_test.json
        |   |   |-- fourk_all.json
        |   |-- roi

        | FACE_DATA_DIR
        |-- IJBB
        |   |-- meta
        |   |   |-- ijbb_face_tid_mid.txt
        |   |-- loose_crop
        |   |   |-- xxx.jpg
        """
    def __init__(self, palm_data_dir, face_data_dir, transform=None):
        self.anno_files = [anno_file.format(dataset) for dataset in palm_datasets]
        self.palm_data_dir = palm_data_dir
        self.face_data_dir = face_data_dir
        self.palm_pth = []
        self.face_pth = []
        self.labels = []
        self.process_anno()
        self.transform = transform if transform else transforms.ToTensor()
    
    def __getitem__(self, index):
        face_img = Image.open(self.face_pth[index]).convert("RGB")
        palm_img = Image.open(self.palm_pth[index]).convert("RGB")
        label = self.labels[index]
        label = label
        if self.transform:
            face_img = self.transform(face_img)
            palm_img = self.transform(palm_img)
        return face_img, palm_img, label

    def __len__(self):
        return len(self.face_pth)

    def get_class_num(self):
        return len(set(self.labels))
    
    def process_anno(self):
        for dataset, anno in zip(palm_datasets, self.anno_files):
            with open(anno, "r") as f:
                lines = f.readlines()
                for line in lines:
                    cont = line.strip().split(" ")
                    face_img_file = cont[0]
                    palm_img_file = f"palm_{cont[1]}.png"
                    self.face_pth.append(os.path.join(self.face_data_dir, "IJBB", "loose_crop", face_img_file))
                    self.palm_pth.append(os.path.join(self.palm_data_dir, dataset, "roi", palm_img_file))
                    self.labels.append(int(cont[3]))

def get_default_transfrom(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])  

if __name__ == "__main__":
    dataset = FusionDataset(
        palm_data_dir="/data/tx/palm_data",
        face_data_dir="/data/tx/IJB",
        transform=get_default_transfrom()
    )
    print(len(dataset))
    print(dataset[0][0].size())
    print(dataset[0][1].size())
    print(dataset[0][2])
        