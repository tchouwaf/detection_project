import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


train_labels_loc = "DataCadot/labels/train"
test_labels_loc = "DataCadot/labels/valid"

def get_labels(path):
    labels_loc = os.path.join(os.getcwd(), path)
    labels = defaultdict(list)

    for root, _, files in os.walk(labels_loc):
        for i, file in enumerate(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        label = parts[0]
                        labels[i].append(label)
    return dict(labels)

y_train_labels = get_labels(train_labels_loc)
y_train_flat_labels = [int(lab) for labs in y_train_labels.values() for lab in labs]

y_test_labels = get_labels(test_labels_loc)
y_test_flat_labels = [int(lab) for labs in y_test_labels.values() for lab in labs]

labels = np.unique(y_test_flat_labels)
print(labels)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(y_train_flat_labels, rwidth=0.8, edgecolor='black')
axes[0].set_title("Distribution des labels (train set)")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Fréquence")

axes[1].hist(y_test_flat_labels, rwidth=0.8, edgecolor='black')
axes[1].set_title("Distribution des labels (test set)")
axes[1].set_xlabel("Label")
axes[1].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()
