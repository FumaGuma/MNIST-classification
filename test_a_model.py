from utils import load_model, evaluate
from data_loader import MnistDataloader

model = load_model("tiny_model1")

mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

train_accuracy = evaluate(x_train, y_train, model)
test_accuracy = evaluate(x_test, y_test, model)
print(model.get_description())
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")
