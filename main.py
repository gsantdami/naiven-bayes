import numpy as np

X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])

Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1 , 0]])

def get_labels_index (labels):
    from collections import defaultdict
    labels_index = defaultdict(list)
    for i, label in enumerate(labels):
        labels_index[label].append(i)
    return labels_index



def get_prior(label_indexes):
    prior = {label: len(indexes) for label, indexes in label_indexes.items()}
    total = sum(prior.values())

    for label in prior:
        prior[label] /= total
    return prior

label_indexes = get_labels_index(Y_train)
prior = get_prior(label_indexes)
print(f"Prior: {prior}")