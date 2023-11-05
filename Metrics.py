import matplotlib.pyplot as plt
import numpy as np

def metrics(Ytrue, Ypred):
    tp, tn, fp, fn, score = 0, 0, 0, 0, 0

    for i in range(len(Ytrue)):
        # if Ytrue[i] == 1 and Ypred[i] == 1:
        #     tp += 1
        # elif Ytrue[i] == 0 and Ypred[i] == 0:
        #     tn += 1
        # elif Ytrue[i] == 0 and Ypred[i] == 1:
        #     fp += 1
        # elif Ytrue[i] == 1 and Ypred[i] == 0:
        #     fn += 1

    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = (2 * precision * recall) / (precision + recall)
        print(np.sqrt(np.matmul(Ytrue[i]-Ypred[i], np.transpose(Ytrue[i]-Ypred[i]))))
        score+= np.sqrt(np.matmul(Ytrue[i]-Ypred[i], np.transpose(Ytrue[i]-Ypred[i])))/(2*len(Ytrue))
    accuracy = 1-score
    precision, recall, f1 = 0, 0, 0

    return accuracy, precision, recall, f1

def plot_metrics(Acc, Loss):

    fig = plt.figure()

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(Acc)
    plt.title('Accuracy')

    plt.show()
    #plt.savefig('AccuracyNLoss.png')

def plot_error_frac(Acc, val_Acc):

    fig = plt.figure()
    plt.xlim([1, len(Acc)])
    plt.plot(np.ones(len(Acc))-Acc, color='b',label="training")
    if val_Acc!=None:
        plt.plot(np.ones(len(val_Acc))-val_Acc, color='g',label="validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title('Error fraction')
    plt.show()
    #plt.savefig('ErrorFrac.png')

def plot_confusion_matrix(prediction, labels, title):
    mat = np.zeros(100).reshape(10, 10)
    for i in range(len(prediction)):
        mat[int(labels[i])][prediction[i]] = mat[int(labels[i])][prediction[i]]+1

    fig, ax = plt.subplots()
    cax = ax.imshow(mat, cmap='viridis', interpolation='nearest')
    cbar = fig.colorbar(cax)
    plt.xlabel("Predictions")
    plt.ylabel("True labels")
    plt.title(title)
    plt.yticks(np.arange(10), np.arange(10))
    plt.xticks(np.arange(10), np.arange(10))
    plt.show()
    #plt.savefig(title)
