import os
import random
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

try:
    from dataloader import preprocess
except ImportError:
    pass

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)

class MyImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))
            dataL = dataL.crop((w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

def get_dataloaders(datapath, batch_size=4, test_batch_size=2):
    left_image_path = os.path.join(datapath, 'left')
    right_image_path = os.path.join(datapath, 'right')
    disparity_path = os.path.join(datapath, 'disparity')

    left_images = sorted([os.path.join(left_image_path, img) for img in os.listdir(left_image_path) if is_image_file(img)])
    right_images = sorted([os.path.join(right_image_path, img) for img in os.listdir(right_image_path) if is_image_file(img)])
    disparity_images = sorted([os.path.join(disparity_path, img) for img in os.listdir(disparity_path) if is_image_file(img)])

    # Train/test split 
    train_size = int(0.8 * len(left_images))
    
    combined = list(zip(left_images, right_images, disparity_images))
    random.shuffle(combined)
    left_images, right_images, disparity_images = zip(*combined)

    train_left_img = left_images[:train_size]
    train_right_img = right_images[:train_size]
    train_left_disp = disparity_images[:train_size]

    test_left_img = left_images[train_size:]
    test_right_img = right_images[train_size:]
    test_left_disp = disparity_images[train_size:]

    TrainImgLoader = torch.utils.data.DataLoader(
        MyImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        MyImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False)

    return TrainImgLoader, TestImgLoader
