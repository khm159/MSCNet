from torch.utils.data import Dataset
from PIL import Image
import os

class ImageAR_Dataset(Dataset):
    def __init__(self, image_list, image_root, transform=None, target_transform=None):

        self.img_list = self._load_txt_from_path(image_list)
        self.image_root = image_root
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_txt_from_path(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        split = self.img_list[idx].split(' ')
        img_name = split[0]
        label = int(split[1])
        img_path = os.path.join(self.image_root, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label