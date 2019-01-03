"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
#Train.py
#trains the classifier and updates model
#calls

from collections import Counter
import numpy as np
import enchant
from scipy import stats
import scipy.linalg
import utils.utils as utils
#-----------------------------------------------------------------------------------------

def reduce_dimensions(feature_vectors_full, model):
    """Dummy methods that just takes 1st 10 pixels.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    selected_features = model['selected_features']

    #projecting the 2340 dimensional images onto the first 40 principal components
    #it is necessary to 'centre' the data before transforming it by subtracting the
    #mean letter vector.
    #perform the reduction -> returns a list
    reduced_fvectors = np.dot((feature_vectors_full - np.mean(feature_vectors_full)),
                              model['principal_comp'])

    #getting the 10 out of 40 principal components

    return reduced_fvectors[:, selected_features]

#-----------------------------------------------------------------------------------------
def principal_components(feature_vectors_full):
    """
    Finds the principal components. Principal components are the eigenvectors of the
    covariance matrix of the train_data of the images

    feature_vectors_full- feature vectors stored as rows in a matrix

    """
    #find the covariance matrix
    #feature_vectors is a (14395,2340) matrix so its covariance will be (2340,2340)
    covx = np.cov(feature_vectors_full, rowvar=0)
    #find the eigenvectors
    N = covx.shape[0] #N is the number of rows of the covariance matrix so its 2340

    #finding the first 40 principal components
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v) #ndarray (2340,40)

    return v

#-----------------------------------------------------------------------------------------
def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    #images is a 2d array [14395][2340]
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width
#-----------------------------------------------------------------------------------------

def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images) #tuple : (39,60)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w #(number of features/dimensions is 2340)
    fvectors = np.empty((len(images), nfeatures)) #empty matrix of feature vectors
    for i, image in enumerate(images):
        #padded image is a 39*60 matrix and each element in the matrix is number 255
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors
#-----------------------------------------------------------------------------------------

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    images_train = [] #initialise empty lists for train data and train labels
    labels_train = []
    for page_name in train_page_names:
        #gets train data from model file using utils.py
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)

    #makes it to numpy array - 14395*1 vector
    labels_train = np.array(labels_train)#numpy array of train labels

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size) #shape: (14395,2340)

    #model data is a dictionary, key value assosciations
    #2 initial keys: labels_train and bbox size
    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size #tuple (39,60)

    print('Extracting principal components')
    #store the principal components from the related function
    p_comp = principal_components(fvectors_train_full)

    #store principal components into the model by making them a list
    model_data['principal_comp'] = p_comp.tolist()

    #store external dictionary text file in the model
    #used for the error correction
    print('Getting the dictionary')
    with open('wiki-100k.txt', 'r') as wiki:
        words = wiki.readlines()
    words_length = [word.strip() for word in words]
    model_data['textFile'] = words_length

    #select the best features
    print("Select Best Features")
    selected_features = select_features(fvectors_train_full, model_data)
    model_data['selected_features'] = selected_features

    #fvectors_train_full has shape (14395, 2340)
    print('Initial dimensions: '+str(fvectors_train_full.shape))

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    #fvectors_train has shape (14395, 10)
    print('Reduced dimensions: '+str(fvectors_train.shape))

    #store fvectors in model_data as list
    model_data['fvectors_train'] = fvectors_train.tolist()

    #return the model
    return model_data
#-----------------------------------------------------------------------------------------

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    print("Loading Test Page")
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)

    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced
#-----------------------------------------------------------------------------------------

def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    print("Classification")

    #select train data and labels
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    #calculate the distance between feature vectors
    #used for classification
    #this distance function gives out the best result
    dist = neg_cosine_distance_multi(page, fvectors_train)

    #this distance function gives out the best result
    labels = knn_classify(fvectors_train, labels_train, page, dist, k=4)


    return np.array(labels)
#-----------------------------------------------------------------------------------------
def find_best_k(train_data, train_labels, test_data, test_labels, labels, dist):
    """
    method for finding the best k. Didn't use it because the score was higher with other k
    so i tried brute forcing.

    Params:
    train_data: feature vectors stored as rows in a matrix
    train_labels: labels for each train data
    test_data: matrix, each row is a feature vector to be classified
    test_labels: page labels
    labels: characters of each page
    dist: distance between the feature vectors

    """
    scores = []
    for k in range(1, 31, 2):
        labels = knn_classify(train_data, train_labels, test_data, dist)
        score = compute_score(test_labels, labels)
        scores[k] = score

    k = scores.index(max(scores))

    return k
#-----------------------------------------------------------------------------------------
def knn_classify(train, train_labels, test, dist, k=1):
    """
    A method for classifying

    Params:
    train: feature vectors stored as rows in a matrix
    train_labels: labels for each train data
    test: matrix, each row is a feature vector to be classified
    dist: distance between the feature vectors
    k = number of neighbours chosen

    """

    knearest = np.argpartition(dist, k, axis=1) [:, 0:k]
    k_labels = train_labels[knearest]
    label, count = stats.mode(k_labels, axis=1)
    label = np.reshape(label, len(test))

    return label


#-----------------------------------------------------------------------------------------
def knn_classify_first_attempt(train, train_labels, test, dist, k=1):
    """
    A method for classifying

    train: feature vectors stored as rows in a matrix
    train_labels: labels for each train data
    test: matrix, each row is a feature vector to be classified
    dist: distance between the feature vectors
    k= number of neighbours chosen
    """
    knearest = np.argsort(dist, axis=1)[:, 0:k]
    k_labels = train_labels[knearest]
    label, count = stats.mode(k_labels, axis=1)
    label = np.reshape(label, len(test))
    return label

#-----------------------------------------------------------------------------------------
def nn_classify(page, fvectors_train, labels_train, dist):
    """
    A method for classifying

    fvectors_train: feature vectors stored as rows in a matrix
    labels_train: labels for each train data
    page: matrix, each row is a feature vector to be classified
    dist: distance between the feature vectors
    dist = number of neighbours chosen
    """
    """Perform nearest neighbour classification using cosine distance. """
    nearest = np.argmin(dist, axis=1)
    label = train_labels[nearest]
    return label
#-----------------------------------------------------------------------------------------
def nn_classify_simple(train, train_labels, test, distance):
    """
    A method for classifying

    train: feature vectors stored as rows in a matrix
    train_labels: labels for each train data
    test: matrix, each row is a feature vector to be classified
    distance: distance between the feature vectors
    """
    """Perform nearest neighbour classification using cosine distance. """
    ntrain = train.shape[0]
    ntest = test.shape[0]
    dist = np.zeros(ntest, ntrain)
    for test_index in range(ntest):
        for train_index in range(ntrain):
            dist[test_index, train_index] = distance(test[test_index, :], train[train_index, :])
    nearest = np.argmin(dist, axis=1)
    label = train_labels[nearest]

    return label
#-----------------------------------------------------------------------------------------
def compute_score(guessed_labels, correct_labels):
    """
    This method was used mostly for testing the classification score
    """
    return 100.0 * sum(guessed_labels == correct_labels) / correct_labels.shape[0]
#-----------------------------------------------------------------------------------------
def euclidean_distance(x, y):
    """
    Method for calculating distance between feature vectors
    """
    """ From the lecture """
    diff = x - y
    return np.sqrt(np.sum(diff * diff))
#-----------------------------------------------------------------------------------------
def cosine_distance_multi(y, x):
    """
    Method for calculating distance between feature vectors
    """
    """ From the labs """
    """ From the labs """
    y_xt = np.dot(y, x.transpose())
    mod_x = np.sqrt(np.sum(x * x, axis=1))
    mod_y = np.sqrt(np.sum(y * y, axis=1))
    return y_xt / np.outer(mod_y, mod_x.transpose())
#-----------------------------------------------------------------------------------------

def neg_cosine_distance_multi(y, x):
    """
    Method for calculating distance between feature vectors
    """
    """ From the lecture """
    y_xt = np.dot(y, x.transpose())
    mod_x = np.sqrt(np.sum(x * x, axis=1))
    mod_y = np.sqrt(np.sum(y * y, axis=1))
    return -y_xt / np.outer(mod_y, mod_x.transpose())

#-----------------------------------------------------------------------------------------
def neg_cosine_distance(x, y):
    """
    Method for calculating distance between feature vectors
    """
    """ From the lecture """
    x_dot_y = np.dot(x, y)
    mod_x = np.sqrt(np.sum(x * x))
    mod_y = np.sqrt(np.sum(y * y))
    return x_dot_y / (mod_x * mod_y)
#-----------------------------------------------------------------------------------------
def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    #find_errors (page, labels, bboxes, model)

    return labels
#-----------------------------------------------------------------------------------------
def find_errors(page, labels, bboxes, model):
    """
    Method for finding and correcting errors. Not used because there was an error.
    """
    print("Error Correction")
    text_file = model['textFile']

    d = enchant.Dict("en_GB")
    words_list, labels_pos_list = construct_words(labels, bboxes)

    #creates an updated words list by replacing non-words with suggestions
    updated_words_list = []
    for word in words_list:
        if not word in text_file:
            sugg_word_list = d.suggest(word)
            #filter the words that have the same length with the word we need to change
            chosen_words_lists = list(filter(lambda x: (len(x) == len(word)), (sugg_word_list)))
            if len(chosen_words_lists) == 0: #if there are no suggestions
                updated_words_list.append(word) #take the word as it is
            else:
                #else choose the first word of the chosen_words_list
                chosen_word = (lambda ch_word: ch_word[0])(chosen_words_lists)
                updated_words_list.append(chosen_word)
        else:
            #if it is in the text file append it to the updated_words_list
            updated_words_list.append(word)


    #check if there is any changed words
    #if there is, change the label using the labels_pos_list positions
    updated_labels = []
    updated_word = []

    for x in range(len(updated_words_list)):
        #if the word is actually changed from the classification word
        if updated_words_list[x] != words_list[x]:
            updated_word = list(updated_words_list[x])
            for y in range(len(labels_pos_list[x])):
                index = int(labels_pos_list[x][y]) #the index of labels to be changed
                labels[index] = updated_word[y] #puts each letter to the label index specified

    for i in range(15):
        print(labels[i])

    return np.array(labels)
#-----------------------------------------------------------------------------------------
def construct_words(labels, bboxes):
    """
    Method for constructing words taking into account the bbox size of each
    letter.
    """
    new_line_change_size = 40
    space_change_size = 7
    word = []
    words_list = []
    label_pos = []
    labels_pos_list = [] #2d array-store the indices of the labels that are changed for each word
    unwanted_chars = [";", ".", ",", "!", "?", ":"]

    for i in range(1, len(labels)):
        space_Change_Bool = ((bboxes[i][0]) - (bboxes[i-1][2])) >= space_change_size
        new_Line_Change_Bool = ((bboxes[i-1][1]) - (bboxes[i][1])) >= new_line_change_size
        word_change_Bool = space_Change_Bool or new_Line_Change_Bool
        if labels[i-1] not in unwanted_chars:
            word += labels[i-1] #constructs a word
            label_pos.append(str(i))
            if (word_change_Bool):
                words_list.append("".join(word))
                labels_pos_list.append(label_pos)
                word = []
                label = []

    return words_list, labels_pos_list
#-----------------------------------------------------------------------------------------
def divergence(class1, class2):
    """compute a vector of 1-D divergences

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores
    """

    #Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12
#-----------------------------------------------------------------------------------------
def multidivergence(class1, class2, features):
    """compute divergence between class1 and class2

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    features - the subset of features to use

    returns: d12 - a scalar divergence score
    """

    ndim = len(features)

    # compute mean vectors
    mu1 = np.mean(class1[:, features], axis=0)
    mu2 = np.mean(class2[:, features], axis=0)

    # compute distance between means
    dmu = mu1 - mu2

    # compute covariance and inverse covariance matrices
    cov1 = np.cov(class1[:, features], rowvar=0)
    cov2 = np.cov(class2[:, features], rowvar=0)

    icov1 = np.linalg.inv(cov1)
    icov2 = np.linalg.inv(cov2)

    # plug everything into the formula for multivariate gaussian divergence
    d12 = (0.5 * np.trace(np.dot(icov1, cov2) + np.dot(icov2, cov1)
                          - 2 * np.eye(ndim)) + 0.5 * np.dot(np.dot(dmu, icov1 + icov2), dmu))

    return d12
#-----------------------------------------------------------------------------------------
def select_features(fvectors_train_full, model):
    """
    Returns a list of best features. Firstly finds the best feature using divergence,
    and then the combinations of features by using multi_divergence
    """
    reduced_fvectors = np.dot((fvectors_train_full - np.mean(fvectors_train_full)), model['principal_comp'])
    labels_train = np.array(model['labels_train'])
    label_sets = list(sorted(set(labels_train)))
    no_of_classes = len(label_sets)

    starting_feature = find_best_feature(reduced_fvectors, label_sets, labels_train, no_of_classes)

    best_features = [starting_feature, 1, 2, 3, 4, 5, 6, 10, 14, 16]

    return best_features
#-----------------------------------------------------------------------------------------

def find_best_feature(train_data, label_sets, labels_train, no_of_classes):
    """
    Finds the feature with the best divergence from all classes
    train_data: matrix - the train data used
    label_sets: matrix - unique sets of features
    labels_train: matrix - the train labels used
    no_of_classes: the length of the label sets which are the different classes
    """

    features = []

    for char1 in range(no_of_classes):
        char1_data = train_data[(label_sets[char1] == labels_train), :]
        for char2 in range(char1 + 1, no_of_classes):
            char2_data = train_data[(label_sets[char2] == labels_train), :]
            div = divergence(char1_data, char2_data)
            f = np.argmax(div)
            features.append(f)

    mode = Counter(np.array(features).reshape(-1)).most_common(1)
    [(best_feature, counts)] = mode

    return int(best_feature)
#-----------------------------------------------------------------------------------------
