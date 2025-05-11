from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name

def load_train_dataset(path: str='c:/PythonProgram/AI/HW3/data/train/')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    label = {0: 'elephant', 1: 'jaguar', 2: 'lion', 3: 'parrot', 4: 'penguin'}
    images = []
    labels = []
    for animal in label:
        subpath = path + label[animal]
        source = os.walk(subpath)
        for root, subfolders, files in source:
        # root: the current path, which is source
        # subfolders: the folders under the root
        # files: files in each folders
            for file in files:
                images.append(subpath+'/'+file)
                labels.append(animal)


    return images, labels

def load_test_dataset(path: str='c:/PythonProgram/AI/HW3/data/test')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    source = os.walk(path)
    for root, subfolders, files in source:
        for file in files:
            if file == 'test_labels.csv':
                continue
            images.append(path+'/'+file)

    return images

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    plt.plot(train_losses, 'g')
    plt.plot(val_losses, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('When Overfitting Happen')
    plt.legend(['train loss', 'validation loss'])
    plt.savefig("loss.png")
    
    print("Save the plot to 'loss.png'")
    return

