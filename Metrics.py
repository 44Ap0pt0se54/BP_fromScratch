# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

        if Ytrue[i] == Ypred[i]:
            score+=1
    accuracy = score/len(Ytrue)
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
