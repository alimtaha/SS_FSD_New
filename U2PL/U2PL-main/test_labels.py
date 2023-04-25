from PIL import Image
from matplotlib import pyplot as plt

plt.imshow(Image.open(
    '/home/extraspace/Datasets/Datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png').convert('L'))
plt.show()
