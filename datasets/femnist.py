import numpy as np
import datasets.np_transforms as tr

from typing import Any
from torch.utils.data import Dataset

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])


class Femnist(Dataset):

    def __init__(self, data: dict, transform: tr.Compose, client_name: str):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name

    def __getitem__(self, index: int) -> Any:
        # TODO: missing code here! -----------> ##### DONE :)
        image = self.samples[index][0]
        image = np.array(image, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

        label = self.samples[index][1]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)
