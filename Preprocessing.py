import numpy as np

def encode(labels, num_classes):
    identity_matrix = np.eye(num_classes)
    one_hot_vectors = []
    for label in labels:
        if label < 0 or label >= num_classes:
            raise ValueError("Label is out of range")
        one_hot_vectors.append(identity_matrix[int(label)])
    return np.array(one_hot_vectors)

def dataSet(imageData, labelData, shuffle):

    n_data_points = 4000
    n_slices = n_data_points // 400  # Number of slices

    train_images = []
    train_labels = []

    for i in range(n_slices):
        start_index = i * 400 + 100
        end_index = start_index + 400
    
        train_images_slice = imageData[start_index:end_index, :]
        train_labels_slice = labelData[start_index:end_index]
    
        train_images.append(train_images_slice)
        train_labels.append(train_labels_slice)

    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    n_data_points = 4000
    n_slices = n_data_points // 100  # Number of slices

    test_images = []
    test_labels = []

    for i in range(n_slices):
        start_index = i * 100
        end_index = start_index + 100
    
        test_images_slice = imageData[start_index:end_index, :]
        test_labels_slice = labelData[start_index:end_index]
    
        test_images.append(test_images_slice)
        test_labels.append(test_labels_slice)

    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    if shuffle:

        shuffle_index = np.random.permutation(1000)
        test_images = test_images[shuffle_index, :]
        test_labels = test_labels[shuffle_index]

        shuffle_index = np.random.permutation(4000)
        train_images = train_images[shuffle_index, :]
        train_labels = train_labels[shuffle_index]

    return train_images, train_labels, test_images, test_labels



