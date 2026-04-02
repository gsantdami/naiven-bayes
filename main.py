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

def get_likelihood(features, prior, smoothing=0):
    likelihood = {}
    for label, index in label_indexes.items():
        likelihood[label] = features[index, :].sum(axis=0) + smoothing
        total = len(index)
        likelihood[label] = likelihood[label] / (total + 2 * smoothing)
    return likelihood


label_indexes = get_labels_index(Y_train)
prior = get_prior(label_indexes)
print(f"Prior: {prior}")

smoothing = 1
likelihood = get_likelihood(X_train, label_indexes, smoothing)

print(f"Likelihood: {likelihood}")