import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

def get_total_number_of_batches(data,batch_size):
    return math.floor(data.shape[0] / batch_size)
def batch_data(data, batch_size):
    ###split into batches function
    parts = data.shape[0] / batch_size
    total_batched_data = math.floor(data.shape[0]/batch_size)*batch_size
    batched_data = np.array_split(data[0:total_batched_data], parts, axis= 0)
    return batched_data

def batch_x_y(x_data, y_data, batch_size):
    batched_x = batch_data(x_data, batch_size)
    batched_y = batch_data(y_data, batch_size)
    return batched_x, batched_y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle_and_batch(x, y, batch_size):
    x, y = unison_shuffled_copies(x, y)
    x, y = batch_x_y(x, y, batch_size)
    return x, y

def save_model(model, filename):
    with open(f"models/{filename}", 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(f"models/{filename}", 'rb') as inp:
        model = pickle.load(inp)
    return model

def evaluate(x_data, y_data, network):
    """Outputs the accuracy for the specified dataset
        x_data : input image
        y_data : labels

        outputs : percentage of correctly classified results
    """
    total = x_data.shape[0]
    x_data, y_data = batch_x_y(x_data, y_data, 50)
    count = 0
    for i in range(len(x_data)):
        x_out = list(network.forward(x_data[i], training = False).argmax(axis=1))
        y_out = list(y_data[i].argmax(axis=1))
        count += sum(x == y for x, y in zip(x_out, y_out))
    return count / total

def get_predictions_and_images(model, x_data, i, n):
    print(np.argmax(model.forward(x_data[i:i + 1, :])))
    #plt.imshow(x_data[i].reshape(28, 28))
    #plt.show()

    #fig, axs = plt.subplots(n, sharex=True, sharey=True)
    fig, axs = plt.subplots(n, n)
    #fig.suptitle('Predictions')
    for j in range(n):
        for k in range(n):
            axs[j, k].imshow(x_data[i + j*n + k].reshape(28, 28))
            axs[j, k].set_title(f"Predicted: {np.argmax(model.forward(x_data[i + j*n + k:i + j*n + k + 1, :]))}")
    fig.tight_layout(pad=0.1)
    plt.show()

class LossTracker():
    def __init__(self, number_of_batches_to_average, epoch_subset_sample_number, batch_size, total_batch_number):
        self.train_loss_for_display = []
        self.train_hits = 0
        self.test_hits = 0
        self.averaged_loss = []
        self.epoch_subset_accuracy_train = []
        self.epoch_subset_accuracy_test = []
        self.number_of_batches_to_average = number_of_batches_to_average
        self.epoch_subset_sample_number = epoch_subset_sample_number
        self.total_batch_number = total_batch_number
        self.batch_size = batch_size
    def add_train_loss(self, loss):
        self.train_loss_for_display.append(loss)
        if len(self.train_loss_for_display) % self.number_of_batches_to_average == 0:
            self.averaged_loss.append(sum(self.train_loss_for_display[-self.number_of_batches_to_average:]) / self.number_of_batches_to_average)
            self.train_loss_for_display = []
    def update_train_accuracy(self, output, y_data):
        number_of_hits = len([i for i, j in zip(output.argmax(axis=1), y_data.argmax(axis=1)) if i == j])
        self.train_hits += number_of_hits

    def update_test_accuracy(self, output, y_data):
        number_of_hits = len([i for i, j in zip(output.argmax(axis=1), y_data.argmax(axis=1)) if i == j])
        self.test_hits += number_of_hits

    def take_epoch_subset_values(self):
        self.epoch_subset_accuracy_train.append(self.train_hits/(self.total_batch_number*self.batch_size))
        self.epoch_subset_accuracy_test.append(self.test_hits/(self.epoch_subset_sample_number*self.batch_size))
        self.train_hits = 0
        self.test_hits = 0
    def latest_loss(self):
        try:
            return self.averaged_loss[-1]
        except IndexError:
            pass

    def show_stats(self):
        """Outputs the graphs of loss and accuracy metrics"""
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(self.epoch_subset_accuracy_train, label="Train:")
        ax1.plot(self.epoch_subset_accuracy_test, label="Test:")
        ax1.legend(loc="upper left")
        ax1.set_xlabel("Number of epochs")
        ax1.set_ylabel("Accuracy")
        ax2.plot(np.log(self.averaged_loss))
        ax2.set_xlabel("Number of batches/100")
        ax2.set_ylabel("Log Loss")
        fig.tight_layout(pad=1.0)
        fig.show()