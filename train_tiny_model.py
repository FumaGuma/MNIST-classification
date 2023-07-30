from data_loader import MnistDataloader
from layers import DenseLayer, NN, ReLu, Sigmoid, Dropout, CrossEntropyLoss
from utils import shuffle_and_batch, evaluate, LossTracker, get_total_number_of_batches, save_model, load_model
from run_model import run_model

"""Load MNIST data
    x_train : train inputs
    y_train : train labels
    x_test : test inputs
    y_test : test labels
 """
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

"""Create a model
    Available layers are Dense, Sigmoid, ReLu and Dropout
"""
model = NN()
model.add_layer(DenseLayer(784,10))

"""Initialize loss function"""
cross_entropy_loss = CrossEntropyLoss()

"""Set parameters
    learning_rate : controls the amount of weight updates per batch
    epochs : number of times model goes through the dataset
    batch_size : size of the input data that is processed simultaneously
    test_batches_n : number of batches that are being run during the training on the train dataset
"""
learning_rate = 5
epochs = 50
batch_size = 64
test_batches_n = 40

model = run_model(model, cross_entropy_loss, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate, test_batches_n)

train_accuracy = evaluate(x_train, y_train, model)
test_accuracy = evaluate(x_test, y_test, model)
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

save_model(model, "tiny_model")
