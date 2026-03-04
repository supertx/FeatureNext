from xtcocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor


def random_error(middle: int, size: int) -> np.ndarray:
    sign = np.random.choice([-1, 1], size=size)
    error = np.random.normal(middle, np.amax([1, middle / 10]), size=size)

    return sign * error


def point_line_distance(line_start: list, line_end: list, point: list) -> float:
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


def gen_roi(
    img: np.ndarray,
    keypoints: np.ndarray,
    roi_size: tuple = (128, 128),
    radius_factor: list = 1.0,
    random_offset: bool = False,
) -> np.ndarray:
    """
    Generate ROI from keypoints
    Args:
        img: (H, W, C)
        keypoints: (N, 2)
        roi_size: tuple[int, int]
        radius_factor: float
    Returns:
        roi: (H', W', C)
    """

    v1 = keypoints[0]
    v2 = keypoints[1]
    v3 = keypoints[2]
    c = keypoints[3]

    error = np.linalg.norm(v1 - v3) * 0.1
    c = np.array(c) + random_error(error, 2) if random_offset else np.array(c)

    # along the finger direction is the vertical axis
    vertical_axis = v2 - c
    horizontal_axis = v1 - v3

    vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
    horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)

    # determine the radius of the palmprint area
    vertical_radius = np.linalg.norm(v2 - c) * 0.5
    d1 = point_line_distance(c, v2, v1)
    d2 = point_line_distance(c, v2, v3)
    horizontal_radius = (d1 + d2) * 0.6

    # determine whether the palmprint area is on the left or right hand
    temp = np.cross(vertical_axis, horizontal_axis)
    is_left = temp < 0
    is_right = not is_left

    # get horizontal axis direction based on left or right hand
    theta = np.pi / 2 if is_right else -np.pi / 2
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    horizontal_axis = rotate_matrix @ vertical_axis

    # adjust the angle of the palmprint area
    angle = 90.0 + np.arctan2(vertical_axis[1], vertical_axis[0]) * 180 / np.pi
    rotation = cv2.getRotationMatrix2D(c.astype(np.float32), angle, 1.0)
    roi = cv2.warpAffine(img, rotation, img.shape[:2][::-1])

    # calculate the radius of the palmprint area
    is_single = False
    if isinstance(radius_factor, float):
        radius_factor = [radius_factor]
        is_single = True

    outs = []
    for factor in radius_factor:
        radius = horizontal_radius * factor

        x_min = max(int(c[0] - radius), 0)
        x_max = min(int(c[0] + radius), roi.shape[1])
        y_min = max(int(c[1] - radius), 0)
        y_max = min(int(c[1] + radius), roi.shape[0])

        # resize the palmprint area
        out = cv2.resize(roi[y_min:y_max, x_min:x_max], roi_size)
        outs.append(out)

    return outs[0] if is_single else outs


def collate_fn(batch):
    item = {}

    rois = [item["roi"] for item in batch]
    labels = [item["label"] for item in batch]
    rois = torch.stack(rois, dim=0)
    labels = torch.tensor(labels)

    item["rois"] = rois
    item["labels"] = labels

    if "comp_roi" in batch[0]:
        comp_rois = [item["comp_roi"] for item in batch]
        comp_rois = torch.stack(comp_rois, dim=0)

        item["comp_rois"] = comp_rois

    return item


class ROIDataset(Dataset):

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform=None,
        roi_size: tuple = (128, 128),
        comp: bool = False,
        radius_factor: list = [1.0],
        use_cache: bool = False,
        random_offset: bool = False,
    ):
        if transform is None:
            transform = T.Compose([
                T.ToTensor(),
                T.Resize((128, 128), antialias=False),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

        self.root_dir = img_dir
        self.roi_size = roi_size
        self.transform = transform
        self.comp = comp
        self.radius_factor = radius_factor
        self.use_cache = use_cache
        self.cache = {}
        self.random_offset = random_offset

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.ids))
        label2ids = {}
        class2label = {}

        for id, ann in zip(self.ids, anns):
            _class = ann["class"]
            if _class not in class2label:
                class2label[_class] = len(class2label)

            _label = class2label[_class]
            if _label not in label2ids:
                label2ids[_label] = []
            label2ids[_label].append(id)

        self.label2ids = label2ids
        self.class2label = class2label
        self.num_classes = len(class2label)

    def __len__(self):
        return len(self.ids)

    def _get_img_data(self, img_id):
        if self.use_cache and img_id in self.cache:
            img, rois, label = self.cache[img_id]
        else:
            anno_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(anno_ids)
            img_path = os.path.join(self.root_dir,
                                    self.coco.loadImgs(img_id)[0]["file_name"])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            keypoints = anns[0]["keypoints"]
            keypoints = np.array(keypoints).reshape(-1, 3)
            keypoints = keypoints[:, :2]
            label = self.class2label[anns[0]["class"]]

            rois = gen_roi(img, keypoints, self.roi_size, self.radius_factor,
                           self.random_offset)

            if self.use_cache and len(self.cache) < 5000:
                self.cache[img_id] = (img, rois, label)

        rois = [self.transform(roi) for roi in rois]
        rois = torch.stack(rois, dim=0)  # (num_rois, C, H, W)

        return rois, label

    def __getitem__(self, idx):
        item = {}

        # train img
        # ========================================================================================
        img_id = self.ids[idx]
        rois, label = self._get_img_data(img_id)
        item["roi"] = rois
        item["label"] = label
        # ========================================================================================

        # comp img
        # ========================================================================================
        if self.comp:
            comp_id = img_id
            while comp_id == img_id and len(self.label2ids[label]) > 1:
                comp_id = np.random.choice(self.label2ids[label])
                comp_id = int(comp_id)  # convert numpy.int64 to int

            comp_rois, comp_label = self._get_img_data(comp_id)
            item["comp_roi"] = comp_rois

        return rois, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    dataset = ROIDataset(
        img_dir="/data/tx/palm_data/IITD/resized",
        ann_file="/data/tx/palm_data/IITD/annotations/fourk_train.json",
        transform=T.ToTensor(),
        comp=False,
        radius_factor=[1.0, 1.25, 1.5],
    )
    data = dataset[0][0][0]
    plt.imshow(data.permute(1, 2, 0).cpu().numpy())
    plt.savefig("1.png")
    # loader = DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4,
    # )
    # rois, labels = next(iter(loader))
    # print(rois.shape, labels.shape)
