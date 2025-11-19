import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_image_train = "DataCadot/images/train"
root_image_test = "DataCadot/images/valid"
root_label_train = "DataCadot/label/train"
root_label_test = "DataCadot/label/valid"

X_train_location = os.path.join(os.getcwd(), root_image_train)
X_test_location = os.path.join(os.getcwd(), root_image_test)

y_train_location = os.path.join(os.getcwd(), root_label_train)
y_test_location = os.path.join(os.getcwd(), root_label_test)

for root, _, files in os.walk(X_train_location):
    images = []
    for file in files :
        # print(f"Root : {root}, Directory = {dir}, files = {file}")
        image = plt.imread(os.path.join(root, file))
        img = np.array(image)
        images.append(img)
        




