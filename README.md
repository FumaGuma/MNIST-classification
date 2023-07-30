Implementation of simple neural network architecture in numpy.

Examples of trained models are in the `models` folder and can be loaded, see `test_a_model.py` file for details.

Tiny model has only one dense layer of size `784, 10` connecting the inputs and outputs, the first version has an exaggerated learning rate.

Medium model consists of tree dense layers + ReLu of size `784, 100`, `100, 50`, `50, 10`. The second medium model introduces a dropout layer which improves the test accuracy.

Big model consists of four dense layers + ReLU of size `784, 512`, `512, 256`, `256, 100`, `100, 10`, it also includes dropout layers and is trained for a larger amount of epochs.

Batch size is `64` and the Cross Entropy Loss is used for all the trained models  .

Results for the example models:

Model | Tiny model 1 | Tiny model 2 | Medium model | Medium model w/ Dropout | Big model w/ Dropout |
--- |--------------|--------------|--------------|-------------------------|----------------------|
Number of epochs | 50           | 50           | 50           | 50                      | 200
Learning rate | 5            | 0.05         | 0.05         | 0.05                    | 0.05
Train accuracy | 91.22%       | 93.04%       | 99.99%       | 99.16%     |   99.77%    
Test accuracy | 90.76%       | 92.47%       | 97.71%       | 98.02%       |  98.71%   
