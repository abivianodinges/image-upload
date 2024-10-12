import torchvision.transforms as transforms
from PIL import Image
import torch
from utils.config import *
import torch.nn as nn
import torch.nn.functional as F

classes = [
        "Mantled Howler",
        "Patas Monkey",
        "Bald Ukari",
        "Japanese Macaque",
        "Pygmy Marmoset",
        "White Headed Capuchin",
        "Silvery Marmoset",
        "Common Squirrel Monkey",
        "Black Headed Night Monkey",
        "Nilgiri Langur"
    ]

def predict_image(jpeg_image):
    
    # Load the model and move it to the appropriate device
    model = torch.load(model_path)
    model.eval()
    
    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((220,220)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    
    # Open the image
    image = Image.open(jpeg_image)
    image = transform(image).float()
    image = image.unsqueeze(0)
    
    # Move the tensor to the same device as the model
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    
    # Make the prediction
    return(classes[predicted.item()])


classes_dog = [
        "dog",
        "notDog"
    ]


def predict_dog(jpeg_image):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 28 * 28, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, len(classes_dog))

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    model = Net()

    model.load_state_dict(torch.load(dog_model_state_path, map_location=device, weights_only=False))
    
    transform = transforms.Compose([
        transforms.Resize(126),
        transforms.CenterCrop(124),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = model.eval()
    image = Image.open(jpeg_image)
    image = transform(image).float()
    image = image.unsqueeze(0)


    # Open the image
    image = Image.open(jpeg_image)
    image = transform(image).float()
    image = image.unsqueeze(0)
    
    # Move the tensor to the same device as the model
    output = model(image)
    _, predicted = torch.max(output.data, 1)

    # Make the prediction
    return(classes_dog[predicted.item()])