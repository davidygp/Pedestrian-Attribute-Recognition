import cv2
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T


class HistogramEqualize():
    def __init__(self, selected_method="heq"):
        self.selected_method = selected_method

    def __call__(self, image):
        # assumes the image is a 3D
        pil_image = image.convert('RGB') 
        open_cv_image = np.array(pil_image) 
        if self.selected_method == "heq":
            heq1 = cv2.equalizeHist(open_cv_image[:,:,0])
            heq2 = cv2.equalizeHist(open_cv_image[:,:,1])
            heq3 = cv2.equalizeHist(open_cv_image[:,:,2])
            img_equalized = np.dstack((heq1, heq2, heq3))
        elif self.selected_method == "clahe":
            clahe = cv2.createCLAHE()
            hcl1 = clahe.apply(open_cv_image[:,:,0])
            hcl2 = clahe.apply(open_cv_image[:,:,1])
            hcl3 = clahe.apply(open_cv_image[:,:,2])
            img_equalized = np.dstack((hcl1, hcl2, hcl3))
        return Image.fromarray(img_equalized)

class UnsharpMask():
    def __init__(self, kernel_size=(5,5), sigma=1.0, a=1.0, cutoff_method="bound"):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.a = a
        self.cutoff_method = cutoff_method

    def __call__(self, image):
        pil_image = image.convert('RGB') 
        open_cv_image = np.array(pil_image) 
        blurred = cv2.GaussianBlur(open_cv_image, self.kernel_size, self.sigma)
        sharpened = (self.a + 1)*open_cv_image - self.a*blurred
        if self.cutoff_method == "rescale":
            # rescale by min_max
            max_val = np.amax(sharpened)
            min_val = np.amin(sharpened)
            sharpened = (sharpened - max_val)/(max_val-min_val)*255
            sharpened = sharpened.round().astype(np.uint8)
        elif self.cutoff_method == "bound":
            # bound by 0 and 255
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            sharpened = sharpened.round().astype(np.uint8)
        return Image.fromarray(sharpened)

class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
	UnsharpMask(kernel_size=(5, 5), sigma=1.0, a=1.0, cutoff_method="bound"),
    	#HistogramEqualize(selected_method="heq")
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
