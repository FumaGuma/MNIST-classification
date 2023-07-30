from utils import get_total_number_of_batches, LossTracker, shuffle_and_batch

def run_model(model, loss_function, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate, test_batches_n):
    import time
    start_tracking = time.time()
    batches_total = get_total_number_of_batches(x_train, batch_size)
    loss_tracker = LossTracker(100, test_batches_n, batch_size, batches_total)
    for k in range(epochs):
        batch_train_x, batch_train_y = shuffle_and_batch(x_train, y_train, batch_size)
        for i in range(len(batch_train_x)):
            print(f"Epoch: {k}, Batch: {i}, Loss: {loss_tracker.latest_loss()}")
            x_data = batch_train_x[i]
            y_data = batch_train_y[i]
            output = model.forward(x_data, training=True)
            ce_loss = loss_function.forward(output, y_data)
            loss_derivative = loss_function.backward()
            loss_tracker.add_train_loss(ce_loss)
            loss_tracker.update_train_accuracy(output, y_data)
            model.backprop(loss_derivative)
            model.update_weights(learning_rate=learning_rate)
        batch_test_x, batch_test_y = shuffle_and_batch(x_test, y_test, batch_size)
        for i in range(test_batches_n):
            x_data = batch_test_x[i]
            y_data = batch_test_y[i]
            output = model.forward(x_data, training=True)
            ce_loss = loss_function.forward(output, y_data)
            loss_tracker.update_test_accuracy(output, y_data)
        loss_tracker.take_epoch_subset_values()
    model.set_training_data(time.time() - start_tracking, learning_rate, epochs, batch_size)
    loss_tracker.show_stats()
    return model