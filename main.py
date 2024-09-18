import time
import argparse
import logging
import matplotlib.pyplot as plt
from splitter import split_file
from preprocessing import create_vocabulary, create_feature_vectors
from perceptron import perceptron_train, error
from analysis import write_important_words

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    start_time = time.time()

    # Split dataset
    split_file(args.spam_train, args.train, args.validation, args.split_line)

    # Create vocabulary and feature vectors
    logging.info("Creating vocabulary and feature vectors.")
    vocabulary = create_vocabulary(args.spam_train, args.threshold)
    x_train, y_train = create_feature_vectors(args.train, vocabulary)
    x_validate, y_validate = create_feature_vectors(args.validation, vocabulary)

    # Train the model and evaluate
    M_values = [200, 400, 800, 1600, 3000]
    validation_errors, epochs_list = [], []

    for M in M_values:
        w, updates, epochs, w_2_norm_history = perceptron_train(x_train, y_train, M=M)
        train_error = error(w, x_train, y_train)
        validation_error = error(w, x_validate, y_validate)
        validation_errors.append(validation_error)
        epochs_list.append(epochs)

        logging.info(f"M = {M}, Updates = {updates}, Epochs = {epochs}, Train Error = {train_error}%, Validation Error = {validation_error}%")

        top_positive_words, top_negative_words = write_important_words(vocabulary, w)
        logging.info(f"Top Positive Words: {top_positive_words}")
        logging.info(f"Top Negative Words: {top_negative_words}")

        plt.plot(range(epochs), w_2_norm_history, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("2-norm of weight vector w")
        plt.title(f"2-norm of weight vector during training, M={M}")
        plt.savefig(f'plots/2_norm_w_vector_M_{M}.png')
        plt.close()

    # Plot overall results
    plot_results(M_values, validation_errors, epochs_list)

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")

def plot_results(M_values, validation_errors, epochs_list):
    plt.figure()
    plt.plot(M_values, validation_errors, marker='o')
    plt.xlabel('M (Number of Training Samples)')
    plt.ylabel('Validation Error (%)')
    plt.title('Validation Error as a Function of M')
    plt.savefig('plots/validation_error_vs_M.png')
    plt.close()

    plt.figure()
    plt.plot(M_values, epochs_list, marker='o')
    plt.xlabel('M (Number of Training Samples)')
    plt.ylabel('Number of Epochs')
    plt.title('Number of Epochs as a Function of M')
    plt.savefig('plots/epochs_vs_M.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron Spam Classification")
    parser.add_argument('--spam_train', type=str, required=True, help="Path to the spam training data file.")
    parser.add_argument('--train', type=str, required=True, help="Path to save the training file.")
    parser.add_argument('--validation', type=str, required=True, help="Path to save the validation file.")
    parser.add_argument('--split_line', type=int, default=3000, help="Line number to split the dataset.")
    parser.add_argument('--threshold', type=int, default=40, help="Threshold for word frequency in vocabulary.")

    args = parser.parse_args()
    main(args)
