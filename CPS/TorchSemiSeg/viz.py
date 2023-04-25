from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
'''
image = Image.open('/media/taha_a/T7/Datasets/cityscapes/city/segmentation/train/aachen_000000_000019_gtFine.png')
im = np.asarray(image)
print(im.shape)
print(image.size)
plt.imshow(im)
plt.colorbar()
plt.show()
'''
mask_shape = (4, 4)
x_prop = np.random.randint(low=0, high=3, size=(5, 3))
y_prop = np.random.randint(low=0, high=3, size=(5, 3))
result = np.stack([y_prop, x_prop], axis=2) * np.array(mask_shape)
result_modif = np.round(result[None, None, :])
positions = np.round(
    (np.array(mask_shape) -
     result_modif) *
    np.random.uniform(
        low=0.0,
        high=1.0,
        size=result_modif.shape))
rectangles = np.append(positions, positions + result_modif, axis=2)


print(np.stack([y_prop, x_prop], axis=2),
      np.stack([y_prop, x_prop], axis=2).shape)

print(
    x_prop,
    "\n",
    y_prop,
    "\n",
    result,
    "\n",
    result_modif.shape,
    result_modif)

print(
    positions,
    "\n",
    positions.shape,
    "\n",
    rectangles,
    "\n",
    rectangles.shape)
