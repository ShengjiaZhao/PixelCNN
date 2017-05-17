from matplotlib import pyplot as plt
import scipy.misc
import numpy as np

img_path = 'samples/long3.jpg'

raw_img = scipy.misc.imread(img_path)
img = np.zeros((raw_img.shape[0] // 28, raw_img.shape[1] // 28, 28, 28, 1))
for i in range(raw_img.shape[0] // 28):
    for j in range(raw_img.shape[1] // 28):
        img[i, j, :, :, 0] = raw_img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28]
print(img.shape)
n_row = 20
n_col = 200
batches = np.reshape(img, (n_row, n_col, 28, 28, 1))

image_list = []
use_index = [9]
for ind in use_index:
    img = batches[ind, :, :, :, :]
    n_row = 10
    n_col = 20
    img = np.reshape(img, (n_row, n_col, 28, 28, 1))
    images = np.zeros((n_row * 28, n_col * 28))
    for i in range(n_row):
        for j in range(n_col):
            images[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img[i, j, :, :, 0]
    image_list.append(images)
image = np.concatenate(image_list, 0)
plt.imshow(1 - image, cmap=plt.get_cmap('Greys'))
plt.show()
scipy.misc.imsave('samples/long3_reshape.jpg', image)
#6  11