import math

def inner_product(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def perceptron_train(x_train, y_train, M=None, max_epochs=None):
    if M is not None:
        x_train = x_train[:M]
        y_train = y_train[:M]

    w = [0] * len(x_train[0])
    updates = 0
    epochs = 0
    w_2_norm_history = []

    while True:
        errors = 0
        for x, y in zip(x_train, y_train):
            if y * inner_product(w, x) <= 0:
                w = [wi + y * xi for wi, xi in zip(w, x)]
                errors += 1
                updates += 1
        w_2_norm_history.append(math.sqrt(inner_product(w, w)))
        epochs += 1
        if errors == 0 or (max_epochs and epochs >= max_epochs):
            break

    return w, updates, epochs, w_2_norm_history

def error(w, x, y):
    error_count = 0
    for i in range(0, len(y)):
        if y[i] * inner_product(w, x[i]) <= 0:
            error_count += 1
    return error_count / len(y) * 100
