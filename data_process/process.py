import os

import torch
from xtcocotools.coco import COCO
from torch.utils.data import Dataset
import pandas as pd 
from random import randint, sample
# import torchvision


datasets = ["IITD", "MPD", "NTU-CP", "SMPD"]
PALM_DATA_DIR = "/data/tx/palm_data"
FACE_DATA_DIR = "/data/tx/IJB"


class PalmDataset(Dataset):
    """
    | PALM_DATA_DIR
    |-- IITD
    |   |-- annotations
    |   |   |-- fourk_train.json
    |   |   |-- fourk_test.json
    |   |   |-- fourk_all.json
    |   |-- resized

    """
    def __init__(
            self, 
            img_root: str = None, 
            train: bool = True, 
            transform = None,
            ):
        super().__init__()
        self.img_root = os.path.join(PALM_DATA_DIR, img_root)
        self.anno_file = os.path.join(PALM_DATA_DIR, img_root, 
                        "annotations", "fourk_all.json" if train else "fourk_test.json")
        self.coco = COCO(self.anno_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        anns = self.coco.loadAnns(self.coco.getAnnIds())

        label2id = {}
        class2label = {}

        for id, ann in zip(self.img_ids, anns):
            _class = ann["class"]
            if _class not in class2label:
                class2label[_class] = len(class2label)

            _label = class2label[_class]
            if _label not in label2id:
                label2id[_label] = []
            label2id[_label].append(id)

        self.label2id = label2id
        self.class2label = class2label
        self.num_classes = len(class2label)

    def get_class_num(self):
        return self.num_classes
    def __len__(self):
        return len(self.img_ids)

def random_ids(class_num, random_ratio):
    rand_num = torch.randn(class_num)
    ids = torch.topk(rand_num, int(class_num * random_ratio)).indices.tolist()
    return ids

def random_from_face_dataset(face_dataset, ids, sample_min_num):
    idx = ids[randint(0, len(ids) - 1)]
    while len(face_dataset.label2id[idx]) <= sample_min_num:
        idx = ids[randint(0, len(ids) - 1)]
    return idx
    

def process(face_dataset):
    face_cls = list(face_dataset.label2id.keys())
    time = 0
    cls_time = 0
    for dataset_name in datasets:
        anno_file = open(f"/home/power/tx/FeatureNext/data_process/result/pro_{dataset_name}_train.txt", "w")
        dataset = PalmDataset(dataset_name)
        cls = random_ids(dataset.get_class_num(), 0.8)
        cls_left = set(range(dataset.get_class_num())) - set(cls)

        # print(f"dataset:{dataset_name}, classes_num: {len(cls)}")
        for label in cls:
            idx = random_from_face_dataset(face_dataset, face_cls, len(dataset.label2id[label]))
            face_cls.remove(idx)
            print(f"{len(face_dataset.label2id[idx])} {len(dataset.label2id[label])}")
            samples = sample(face_dataset.label2id[idx], len(dataset.label2id[label]))
            for id_f, id_p in zip(samples, dataset.label2id[label]):
                anno_file.write(f"{id_f} {id_p} {time} {cls_time}\n")  
                time += 1
            cls_time += 1
        anno_file.close()
        
        anno_file = open(f"/home/power/tx/FeatureNext/data_process/result/pro_{dataset_name}_cls_left.txt", "w")
        anno_file.write(list(cls_left).__str__() + "\n")
        anno_file.close()

    anno_file = open(f"/home/power/tx/FeatureNext/data_process/result/pro_face_cls_left.txt", "w")
    anno_file.write(face_cls.__str__() + "\n")
    anno_file.close()
        # for label in cls_left:
        #     idx = random_from_face_dataset(face_dataset, face_cls, len(dataset.label2id[label]))
        #     face_cls.remove(idx)
        #     samples = sample(face_dataset.label2id[idx], len(dataset.label2id[label]))
        #     for id_f, id_p in zip(samples, dataset.label2id[label]):
        #         anno_file.write()
        



class IJBDataset(Dataset):
    """
    | DATA_DIR
    |-- IJBB
    |   |-- meta
    |   |   |-- ijbb_face_tid_mid.txt
    |   |-- loss_crop
    |   |   |-- xxx.jpg
    """
    def __init__(self, img_root, train=True):
        super().__init__()
        self.label2id = {}
        self.img_labels = []
        self.min_palm_img_num = float('inf')
        anno_file_path = os.path.join(FACE_DATA_DIR, img_root, "meta", "ijbb_face_tid_mid.txt")
        self.process_anno(anno_file_path)
        self.img_root = os.path.join(FACE_DATA_DIR, img_root, "loss_crop")
    def process_anno(self, anno_file_path):
        with open(anno_file_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip().split(" ")
            img_id = line[0]
            label = line[1]
            if label not in self.label2id:
                self.label2id[label] = []
            self.label2id[label].append(img_id)
            self.img_labels.append(label)
            
        
    def get_class_num(self):
        return len(self.label2id)
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.img_labels)   





if __name__ == "__main__":
    # class_sum = 0
    # img_num = 0 
    # min_palm_img_num = float('inf')
    # max_palm_img_num = 0
    # for dataset in datasets:
    #     dataset = PalmDataset(dataset)
    #     print(dataset.label2id)
    #     for ids in dataset.label2id.values():
    #         min_palm_img_num = min(min_palm_img_num, len(ids))
    #         max_palm_img_num = max(max_palm_img_num, len(ids))
    #     class_sum += dataset.get_class_num()
    #     img_num += len(dataset)
    face_dataset = IJBDataset("IJBB")
    # print("-------------------------")
    # print("palm")
    # print("class_sum: {}, img_num: {}, min_palm_img_num: {}, max_palm_img_num: {} avg_img_per_class: {}".format(class_sum, img_num, min_palm_img_num, max_palm_img_num, img_num // class_sum))
    # print("-------------------------")
    # print("face")
    # print("class_num: {}, img_num: {}, min_palm_img_num: {}, avg_img_per_class: {}".format(face_dataset.get_class_num(), len(face_dataset), face_dataset.min_palm_img_num, len(face_dataset) // face_dataset.get_class_num()))
    process(face_dataset)