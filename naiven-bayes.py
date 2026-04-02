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

def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        
        sum_posterior = sum(posterior.values())
        for label in posterior: 
            if posterior[label] == float('inf'):
                posterior[label] = 1.0

            else:
                posterior[label] /= sum_posterior
        posteriors.append((posterior.copy()))
    return posteriors



label_indexes = get_labels_index(Y_train)
prior = get_prior(label_indexes)

smoothing = 1
likelihood = get_likelihood(X_train, label_indexes, smoothing)


posterior = get_posterior(X_test, prior, likelihood)
print(f"POSTERIOR: {posterior}")