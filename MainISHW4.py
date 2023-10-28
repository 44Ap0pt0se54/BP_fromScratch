from Input import*
from Sequential import*
from Metrics import*
import matplotlib.pyplot as plt
from PIL import Image
from Preprocessing import*
import numpy as np

# DATA LOADING
imageData = np.loadtxt("MNISTnumImages5000_balanced.txt")
labelData = np.loadtxt("MNISTnumLabels5000_balanced.txt")

# 0 to 9 EXTRACTION + SHUFFLING

train_images, train_labels, test_images, test_labels = dataSet(imageData, labelData, shuffle=True)

############# MODEL ###############

model = Sequential()

model.add(Input(784))
l1 = Dense(50, "relu")
l2 = Dense(10, "softmax")
model.add(l1)
model.add(l2)


model.compile()

############### TRAINING ################

lr = 0.01
model.fit(train_images, train_labels, eta = lr, num_epochs = 5, batch_size = None, loss_type = "CategoricalCrossEntropy", plot=True)

accuracy, precision, recall, f1 = metrics(test_labels, [np.argmax(arr) for arr in model.predict(test_images)])
print(accuracy)

########### HEATMAP ###############

# image_vector = l1.w[:, 1].reshape(28, 28)

# image_matrix = (image_vector * 255).astype(np.uint8)

# # Create a PIL Image
# image = Image.fromarray(image_matrix)

# # Save the image as a PNG file
# image.save("HeatMap01.png")