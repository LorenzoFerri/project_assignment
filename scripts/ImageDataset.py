import os
import os.path
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io
from skimage.transform import resize


class ImageDataset(Dataset):

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = np.array([f for f in os.listdir(
            root_dir) if os.path.isfile(os.path.join(root_dir, f))])
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(os.path.join(self.root_dir, img_name))
        pts1 = np.float32([[280, 0], [360, 0], [0, 480], [640, 480]])
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, matrix, (640, 480))
        params = img_name.split('.png')[0].split('_')
        distance = float(params[1]) * 10
        angle = float(params[2]) * 10
        #image = image.transpose((1, 2, 0))
        image = self.transform(image)
        target = torch.FloatTensor([distance, angle])
        sample = (image, target, img_name)

        return sample
