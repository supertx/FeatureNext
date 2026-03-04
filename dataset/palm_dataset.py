"""
Author: supermantx
Date: 2025-06-10 10:28:59
LastEditTime: 2025-06-11 14:44:35
Description:
"""

import os

from einops import rearrange
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from xtcocotools.coco import COCO
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
SHOW_PLOT = False

colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Cyan
]


def random_error(middle: int, size: int) -> np.ndarray:
    sign = np.random.choice([-1, 1], size=size)
    error = np.random.normal(middle, np.amax([1, middle / 10]), size=size)

    return sign * error


def point_line_distance(line_start: list, line_end: list,
                        point: list) -> float:
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array(point)

    line_vector = line_end - line_start
    line_length = np.linalg.norm(line_vector)
    line_vector = line_vector / line_length

    point_vector = point - line_start
    point_projection = np.dot(point_vector, line_vector)

    projection_point = line_start + point_projection * line_vector
    distance = np.linalg.norm(point - projection_point)

    return distance


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs.
    """
    rois, labels = zip(*batch)
    rois = torch.stack(rois, dim=0)  # Stack the ROIs into a tensor
    # raw_rois = torch.stack([roi[0] for roi in rois], dim=0)
    labels = torch.tensor(labels,
                          dtype=torch.long)  # Convert labels to a tensor
    labels = labels.repeat_interleave(rois[0].shape[0])
    rois = rearrange(rois, 'b n c h w -> (b n) c h w')
    return rois, labels


class PalmDataset(Dataset):

    def __init__(
        self,
        img_root: str,
        anno_file: str,
        transform: transforms = None,
        roi_size: tuple = (128, 128),
        scale_disturb: bool = False,
        center_point_disturb: bool = False,
        angle_disturb: bool = False,
        aug_within_sample: int = 0,
    ):
        super().__init__()

        self.img_root = img_root
        self.anno_file = anno_file
        self.roi_size = roi_size
        self.aug_within_sample = aug_within_sample
        self.disturb_dict = {
            "scale_disturb": scale_disturb,
            "center_point_disturb": center_point_disturb,
            "angle_disturb": angle_disturb,
        }
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            ])
        else:
            self.transform = transform
        self.coco = COCO(anno_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids))

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

    def __get_img_data(self, img_id):
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        annos = self.coco.loadAnns(anno_ids)
        img_path = os.path.join(self.img_root,
                                self.coco.loadImgs(img_id)[0]["file_name"])
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        keypoints = annos[0]["keypoints"]
        keypoints = np.array(keypoints).reshape(-1, 3)
        keypoints = keypoints[:, :2]

        label = self.class2label[annos[0]["class"]]
        return img, keypoints, label

    @staticmethod
    def __gen_roi(
        img: np.ndarray,
        keypoints: np.ndarray,
        roi_size: tuple = (256, 256),
        center_point_disturb: bool = False,
        angle_disturb: bool = False,
        scale_disturb: bool = False,
    ):
        """
        based on myx's method
        """
        v1 = keypoints[0]
        v2 = keypoints[1]
        v3 = keypoints[2]
        c = keypoints[3]

        if center_point_disturb:
            error = np.linalg.norm(v1 - v3) * 0.1
            c = np.array(c) + random_error(error, 2)
        c = np.array(c)
        # determine the radius of the palmprint area
        vertical_axis = v2 - c
        vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
        d1 = point_line_distance(c, v2, v1)
        d2 = point_line_distance(c, v2, v3)
        horizontal_radius = (d1 + d2) * 0.6

        # adjust the angle of the palmprint area
        angle = 90.0 + np.arctan2(vertical_axis[1],
                                  vertical_axis[0]) * 180 / np.pi
        if angle_disturb:
            angle += np.random.randint(-10, 10)
        rotation = cv.getRotationMatrix2D(c.astype(np.float32), angle, 1.0)
        roi = cv.warpAffine(img, rotation, img.shape[:2][::-1])

        if scale_disturb:
            scale = 1.25 + np.random.uniform(-0.25, 0.3)
        else:
            scale = 1.25
        radius = horizontal_radius * scale

        x_min = max(int(c[0] - radius), 0)
        x_max = min(int(c[0] + radius), roi.shape[1])
        y_min = max(int(c[1] - radius), 0)
        y_max = min(int(c[1] + radius), roi.shape[0])

        # resize the palmprint area
        roi = cv.resize(roi[y_min:y_max, x_min:x_max], roi_size)
        return roi

    def __getitem__(self, index):
        img, keypoint, label = self.__get_img_data(self.img_ids[index])
        raw_roi = self.__gen_roi(
            img,
            keypoint,
        )
        dis_roi = [
            self.__gen_roi(
                img,
                keypoint,
                **self.disturb_dict,
            ) for _ in range(self.aug_within_sample)
        ]
        if len(dis_roi) > 0:
            rois = [raw_roi] + dis_roi
            rois = np.stack(rois, axis=0)
            rois = [self.transform(roi) for roi in rois]
            rois = torch.stack(rois, dim=0)
        else:
            rois = self.transform(raw_roi)

        return rois, label

    def __len__(self):
        return len(self.img_ids)

    def visualize(self, index):
        img, keypoint, _ = self.__get_img_data(self.img_ids[index])
        # for i, point in enumerate(keypoint):
        #     cv.circle(img, tuple(point.astype(int)), 3, colors[i], -1)
        plt.imshow(img)
        plt.savefig(f"palm_{index}.png")
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(self.__gen_roi(
            img,
            keypoint,
            self.roi_size,
        ))
        plt.subplot(1, 3, 2)
        plt.imshow(
            self.__gen_roi(
                img,
                keypoint,
                self.roi_size,
                **self.disturb_dict,
            ))
        plt.subplot(1, 3, 3)
        plt.imshow(
            self.__gen_roi(
                img,
                keypoint,
                self.roi_size,
                **self.disturb_dict,
            ))
        plt.savefig(f"palm_roi_{index}.png")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    palm_datasets = ["IITD", "MPD", "NTU-CP", "SMPD"]
    PALM_DATA_DIR = "/data/tx/palm_data"
    for dataset_name in palm_datasets:
        
        dataset = PalmDataset(
            img_root=f"/data/tx/palm_data/{dataset_name}/resized",
            anno_file=f"/data/tx/palm_data/{dataset_name}/annotations/fourk_all.json",
            roi_size=(112, 112),
            aug_within_sample=0,
            transform=T.Compose([
            ]),
        )
        for i, data in zip(range(len(dataset)), dataset):
            os.makedirs(os.path.join(PALM_DATA_DIR, dataset_name, "roi"), exist_ok=True)
            cv.imwrite(os.path.join(PALM_DATA_DIR, dataset_name, "roi", f"palm_{i+1}.png"),
                       cv.cvtColor(data[0], cv.COLOR_RGB2BGR),
                       [cv.IMWRITE_PNG_COMPRESSION, 0])

    # loader = DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4,
    # )
    # rois, labels = next(iter(loader))
    # print(rois.shape, labels.shape)
 