import os
import numpy as np
import matplotlib.pyplot as plt

images_train_loc = "DataCadot/images/train"

total_images_train_loc = os.path.join(os.getcwd(), images_train_loc)

def plot_few_images(path):
    images = []
    for root, _, files in os.walk(path):
        for i, file in enumerate(files[:3]):
            location = os.path.join(root, file)
            image = plt.imread(location)
            img = np.array(image)
            images.append(img)

    return images

images = plot_few_images(total_images_train_loc)

plt.figure()
plt.imshow(images[0])
plt
plt.show()