import cv2
import os
import pickle
import json

import copy
import math
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

class RandomErase():
    def __init__(self, Wr=(0.05, 0.3), Hr=(0.05, 0.3), debug=False):
        self.Wr = Wr
        self.Hr = Hr
        self.debug = debug

    def __call__(self, image):
        pil_image = image.convert('RGB') 
        open_cv_image = np.array(pil_image) 

        H, W, _ = open_cv_image.shape
        image_erased = copy.deepcopy(open_cv_image)

        xe = np.random.randint(0, W)           # random x 
        ye = np.random.randint(0, H)           # random y
        We = np.random.uniform(self.Wr[0], self.Wr[1])*W # random % of width 
        He = np.random.uniform(self.Hr[0], self.Hr[1])*H # random % of height

        xeWef = math.floor(min(xe+We,W))
        yeHef = math.floor(min(ye+He,H))

        if self.debug: 
            print("Total image is %s width by %s height" %(W, H))
            print("Erasing from width %s to %s, and height %s to %s" %(xe, xeWef, ye, yeHef))
        image_erased[ye:yeHef, xe:xeWef, :] = np.random.randint(0, 255)
        return Image.fromarray(image_erased)

class Mosaic():
    def __init__(self, Prob, Wj=0.2, Hj=0.2, debug=False):
        self.Wj = Wj       # Width jitter around the middle
        self.Hj = Hj       # Heigh jitter around the middle
        self.debug = debug # Whether to debug or not 
        self.Prob = Prob

    def __call__(self, images):
        open_cv_images = [np.array(x.convert('RGB')) for x in images]

        largest_image_index = np.array([x.size for x in open_cv_images]).argmax()

        # Use the biggest picture to fit the mosaic
        Hmax, Wmax, _ = open_cv_images[largest_image_index].shape

        x_middle = Wmax/2
        x_join = math.floor(x_middle + np.random.uniform(-self.Wj, self.Wj)*x_middle)
        y_middle = Hmax/2
        y_join = math.floor(y_middle + np.random.uniform(-self.Hj, self.Hj)*y_middle)

        canvas = copy.deepcopy(open_cv_images[largest_image_index])

        if len(open_cv_images) >= 5:
            print("ERROR")
        elif len(open_cv_images) == 4:
            canvas[0:y_join, 0:x_join, :] = cv2.resize(open_cv_images[0], (x_join,y_join), interpolation = cv2.INTER_AREA)
            canvas[0:y_join, x_join:Wmax, :] = cv2.resize(open_cv_images[1], (Wmax-x_join,y_join), interpolation = cv2.INTER_AREA)
            canvas[y_join:Hmax, 0:x_join, :] = cv2.resize(open_cv_images[2], (x_join,Hmax-y_join), interpolation = cv2.INTER_AREA)
            canvas[y_join:Hmax, x_join:Wmax, :] = cv2.resize(open_cv_images[3], (Wmax-x_join,Hmax-y_join), interpolation = cv2.INTER_AREA)
        elif len(open_cv_images) == 3:
            canvas[0:y_join, 0:x_join, :] = cv2.resize(open_cv_images[0], (x_join,y_join), interpolation = cv2.INTER_AREA)
            canvas[y_join:Hmax, 0:x_join, :] = cv2.resize(open_cv_images[1], (x_join,Hmax-y_join), interpolation = cv2.INTER_AREA)
            canvas[0:Hmax, x_join:Wmax, :] = cv2.resize(open_cv_images[2], (Wmax-x_join,Hmax), interpolation = cv2.INTER_AREA)
        elif len(open_cv_images) == 2:
            canvas[0:Hmax, 0:x_join, :] = cv2.resize(open_cv_images[0], (x_join,Hmax), interpolation = cv2.INTER_AREA)
            canvas[0:Hmax, x_join:Wmax, :] = cv2.resize(open_cv_images[1], (Wmax-x_join,Hmax), interpolation = cv2.INTER_AREA)
        return Image.fromarray(canvas)
    
class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PA100k', 'RAP', 'RAP2', 'PETA+PA100k+RAP', 'PETA+RAP', 'PETA+PA100k'], \
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
        #UnsharpMask(kernel_size=(5, 5), sigma=1.0, a=1.0, cutoff_method="bound"),
        #HistogramEqualize(selected_method="heq"),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

def parse_transformation_dict(transformation_dict):
    """
        Parses a transformation dictionary into a list of ordered transformations, ONLY for single image transformations.
        To be used with T.Compose([<...>]) subsequently
    """
    if "Order" not in transformation_dict.keys():
        return [T.ToTensor()]
    all_transformations = []
    for transformation in transformation_dict["Order"]:
        if transformation == None:
            break
        elif transformation == "Resize":
            all_transformations.append(T.Resize(**transformation_dict["Resize"]))
        elif transformation == "Pad":
            all_transformations.append(T.Pad(**transformation_dict["Pad"]))
        elif transformation == "RandomCrop":
            all_transformations.append(T.RandomCrop(**transformation_dict["RandomCrop"]))
        elif transformation == "RandomHorizontalFlip":
            all_transformations.append(T.RandomHorizontalFlip(**transformation_dict["RandomHorizontalFlip"]))
        elif transformation == "ToTensor":
            all_transformations.append(T.ToTensor())
        elif transformation == "Normalize":
            all_transformations.append(T.Normalize(**transformation_dict["Normalize"]))
        elif transformation == "RandomAffine":
            all_transformations.append(T.RandomAffine(**transformation_dict["RandomAffine"]))
        # Self added Classes below, BEWARE
        elif transformation == "UnsharpMask":
            all_transformations.append(UnsharpMask(**transformation_dict["UnsharpMask"]))
        elif transformation == "HistogramEqualize":
            all_transformations.append(HistogramEqualize(**transformation_dict["HistogramEqualize"]))
        elif transformation == "RandomErase":
            all_transformations.append(RandomErase(**transformation_dict["RandomErase"]))
    return all_transformations
    
class AttrDataset_new(data.Dataset):
    """
        Combination of AttrDataset with get_transform
    """

    def __init__(self, split, args, transformation_dict):

        assert args.dataset in ['PETA', 'PA100k', 'RAP', 'RAP2', 'PETA+PA100k+RAP', 'PETA+RAP', 'PETA+PA100k'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transformation_dict = transformation_dict
        self.transform = T.Compose(parse_transformation_dict(self.transformation_dict))
        print(self.transform)
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
        gt_label = gt_label.astype(np.float32)

        if self.transform is not None:
            if "Mosiac" in self.transformation_dict["Order"]:
                p = np.random.rand()
                if p <= self.transformation_dict["Mosaic"]["Prob"] and index >= 3:
                    gt_label_m0 = self.label[index].astype(np.float32)
                    gt_label_m1 = self.label[index-1].astype(np.float32)
                    gt_label_m2 = self.label[index-2].astype(np.float32)
                    gt_label_m3 = self.label[index-3].astype(np.float32)
                    
                    img_m0 = Image.open(os.path.join(self.root_path, self.img_id[index]))
                    img_m1 = Image.open(os.path.join(self.root_path, self.img_id[index-1]))
                    img_m2 = Image.open(os.path.join(self.root_path, self.img_id[index-2]))
                    img_m3 = Image.open(os.path.join(self.root_path, self.img_id[index-3]))
                    
                    imgs = [img_m0, img_m1, img_m2, img_m3]
                    
                    Mosiactransform = T.Compose([Mosaic(**self.transformation_dict["Mosaic"])])
                    img = Mosiactransform(imgs)
                    
                    combined = gt_label_m0 + gt_label_m1 + gt_label_m2 + gt_label_m3
                    gt_label = np.where(combined > 0.5, 1, 0)
                    gt_label = gt_label.astype(np.float32)
                    
                    imgname = "Mosaic Combined"
                img = self.transform(img)      
                if "LabelSmoothing" in self.transformation_dict["Order"]:
                    pos_val = self.transformation_dict["LabelSmoothing"]["pos_val"]
                    assert(pos_val <= 1.0 and pos_val >= 0)
                    gt_label1 = copy.deepcopy(gt_label)
                    gt_label = np.concatenate([gt_label1[0:4], np.where(gt_label1[4:] > 0.5, pos_val, 1-pos_val)])
            else:  
                img = self.transform(img)
        
        if "LabelSmoothing" in self.transformation_dict["Order"] and "Mosiac" not in self.transformation_dict["Order"]:
            pos_val = self.transformation_dict["LabelSmoothing"]["pos_val"] 
            assert(pos_val <= 1.0 and pos_val >= 0)
            gt_label = np.where(gt_label > 0.5, pos_val, 1-pos_val)

        #if self.target_transform is not None:
        #    gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)