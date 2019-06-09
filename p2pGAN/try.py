import os
from PIL import Image
import torchvision.transforms as transforms
dataroot='./cifar10/images_test'
for filename in os.listdir(dataroot):
    img = Image.open(dataroot+'/'+filename)
    img = transforms.ToTensor()(img).cuda()
    