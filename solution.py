import numpy as np
from numpy.linalg import inv

def image2features(image, params):
    """ Computes features for one image """

    # TODO: Use params for PCA etc

    # TODO: Make useful features
    return [
        image[0], # first feature
        image[1] # second feature
    ]

def compute_features(images):
    """ Computes the features for a list of images. """

    # TODO: Compute parameters for features
    params = None

    return np.array([
        image2features(images[i,:], params) for i in range(images.shape[0])
    ])


def train_W(A, B):
    """ Trains the weight matrix W """

    # B is the input (feature vector)
    # A is the output (class labels)

    return np.dot(inv(np.dot(B.T, B)), np.dot(B.T, A))

def compute_labels(images):
    """ Computes the labels (output) for a set of images """

    labels = np.zeros((images.shape[0], 10))
    for c in range(10):
        labels[c*100:(c+1)*100,c] = 1
    return labels

def classify(images, W):
    """ Classifies a set of images """

    # compute features
    features = compute_features(images)

    # apply weights
    weighted = np.dot(features, W)

    # compute the classes, take the maximal element
    return np.argmax(weighted, axis=1)

def compute_should_output(n_elems):
    """ Computes the output that the classifier should produce when having n_elems / class """

    out = np.zeros(n_elems*10)
    for i in range(10):
        out[100*i:100*(i+1)] = i
    return out

def compute_miss_rate(has, should):
    """ Computes the missclassification rate """

    return np.count_nonzero(has - should) / has.shape[0]




def main():
    """ Main function """

    #
    # READ AND PROCESS DATA
    #

    # load the file
    data = np.loadtxt("mfeat-pix.txt")

    # split into test and training data
    train_data = np.vstack([data[200*i:200*i + 100,:] for i in range(10)])
    test_data = np.vstack([data[200*i + 100:200*(i + 1), :] for i in range(10)])

    #
    # TRAINING
    #

    # compute training features and labels (output)
    feature_train = compute_features(train_data)
    labels = compute_labels(train_data)

    # Compute W, see formula in lecture notes
    W = train_W(labels, feature_train)

    #
    # TESTING
    #
    should_out = compute_should_output(100)
    train_out = classify(train_data, W)
    test_out = classify(test_data, W)

    print("Training miss rate", compute_miss_rate(train_out, should_out))
    print("Test miss rate", compute_miss_rate(test_out, should_out))


if __name__ == "__main__":
    main()
