from data_loader import MnistDataloader
from layers import DenseLayer, NN, ReLu, Dropout, CrossEntropyLoss
from utils import evaluate, save_model
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
model.add_layer(Dropout(0.2))
model.add_layer(DenseLayer(784,512))
model.add_layer(ReLu())
model.add_layer(Dropout(0.5))
model.add_layer(DenseLayer(512,256))
model.add_layer(ReLu())
model.add_layer(Dropout(0.3))
model.add_layer(DenseLayer(256,100))
model.add_layer(ReLu())
model.add_layer(DenseLayer(100,10))

"""Initialize loss function"""
cross_entropy_loss = CrossEntropyLoss()

"""Set parameters
    learning_rate : controls the amount of weight updates per batch
    epochs : number of times model goes through the dataset
    batch_size : size of the input data that is processed simultaneously
    test_batches_n : number of batches that are being run during the training on the train dataset
"""
test_batches_n = 10
batch_size = 64
epochs = 200
learning_rate = 0.05
model = run_model(model, cross_entropy_loss, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate, test_batches_n)

train_accuracy = evaluate(x_train, y_train, model)
test_accuracy = evaluate(x_test, y_test, model)
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")
save_model(model, "big_dropout_model")