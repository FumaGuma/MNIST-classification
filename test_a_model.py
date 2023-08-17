from utils import load_model, evaluate
from data_loader import MnistDataloader

from utils import get_predictions_and_images

model = load_model("big_dropout_model")

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

train_accuracy = evaluate(x_train, y_train, model)
test_accuracy = evaluate(x_test, y_test, model)
print(model.get_description())
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

get_predictions_and_images(model, x_test, 160, 3)
