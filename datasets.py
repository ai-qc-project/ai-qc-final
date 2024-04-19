import torch
import cv2
import numpy as np
import os
import glob as glob
import random

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.general import visualize_mosaic_images
from utils.transforms import (
    get_train_transform, get_valid_transform,
    get_train_aug
)

class CustomDataset(Dataset):
    def __init__(
        self, images_path, labels_path, 
        width, height, classes, transforms=None, 
        use_train_aug=False,
        train=False, mosaic=False
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.mosaic = mosaic
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []
        
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        self.read_and_clean()
        
    def read_and_clean(self):
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present == False:
                image_name = annot_path.split(os.path.sep)[-1].split('.xml')[0]
                image_root = self.all_image_paths[0].split(os.path.sep)[:-1]
                # remove_image = f"{'/'.join(image_root)}/{image_name}.jpg"
                remove_image = os.path.join(os.sep.join(image_root), image_name+'.jpg')
                print(f"Removing {annot_path} and corresponding {remove_image}")
                self.all_annot_paths.remove(annot_path)
                self.all_image_paths.remove(remove_image)

        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, image_name.split('.jpg')[0]+'.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")
                self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)
        
        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            ymax, xmax = self.check_image_and_annotation(
                xmax, ymax, image_width, image_height
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmax, ymax, width, height):
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        return ymax, xmax


    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        image, _, _, _, _, _, _, _ = self.load_image_and_labels(index=index)
        orig_image = image.copy()
        image = cv2.resize(image, resize_factor)
        h, w, c = image.shape
        s = h // 2
    
        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]
        indexes = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]
        
        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indexes):
            image, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
            index=index
            )

            image = cv2.resize(image, resize_factor)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h 
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh
            
            result_boxes.append(boxes)
            for class_name in labels:
                result_classes.append(class_name)

        final_classes = []
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        for idx in range(len(result_boxes)):
            if ((result_boxes[idx,2]-result_boxes[idx,0])*(result_boxes[idx,3]-result_boxes[idx,1])) > 0:
                final_classes.append(result_classes[idx])
        result_boxes = result_boxes[
            np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
        ]
        return orig_image, result_image/255., torch.tensor(result_boxes), \
            torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):
        if not self.mosaic:
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and self.mosaic:
            while True:
                image, image_resized, boxes, labels, \
                    area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                    idx, resize_factor=(self.height, self.width)
                )
                if len(boxes) > 0:
                    break

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if self.use_train_aug:
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
