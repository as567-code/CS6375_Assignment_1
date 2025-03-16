import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):

        # Compute the sequence in the first layer.
        first_layer = self.W1(input_vector)
    
        # Convert the output to zero if the output is negative and left unchanged if positive to the hidden state.
        hidden_state = self.activation(first_layer)

        # To change the hidden state to 5 sentiment classes output.
        output_layer = self.W2(hidden_state)
    
        # Applying softmax to convert them into a probability distribution
        predicted_vector = self.softmax(output_layer)
    
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

def plot_learning_curve(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss', markersize=5)  # Blue line with dots
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy', markersize=5)  # Red line with dots
    plt.title('Training Loss and Validation Accuracy (FFNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve.png')  # Save the plot as a PNG image
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))
    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    corresponding_train_accuracy = 0
    best_val_time = 0
    errors = []  # List to store error examples

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector.unsqueeze(0))  # Add batch dimension
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss
                
                # Collect error examples
                if predicted_label != gold_label:
                    errors.append({
                        'input': ' '.join([index2word[i.item()] for i in torch.nonzero(input_vector)]),  # Original text
                        'predicted': predicted_label.item(),
                        'actual': gold_label
                    })
            loss = loss / minibatch_size
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / (N // minibatch_size)
        train_losses.append(avg_loss)
        train_accuracy = correct / total
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector.unsqueeze(0))  # Add batch dimension
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss = example_loss if loss is None else loss + example_loss
            loss = loss / minibatch_size
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_time = time.time() - val_start_time
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_accuracy))
        print("Validation time for this epoch: {}".format(val_time))

        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            corresponding_train_accuracy = train_accuracy
            best_val_time = val_time

    print("Best Validation Accuracy: {:.4f}".format(best_val_accuracy))
    print("Corresponding Training Accuracy: {:.4f}".format(corresponding_train_accuracy))
    print("Best Validation Time: {:.2f} seconds".format(best_val_time))

    # To get training losses and validation accuracies graph
    with open('results.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_accuracies': val_accuracies}, f)

    # To Plot learning curve
    epochs = list(range(1, args.epochs + 1))
    plot_learning_curve(epochs, train_losses, val_accuracies)

    # To get error distribution graph
    with open('errors.json', 'w') as f:
        json.dump(errors, f)

    print("Error examples saved to errors.json")
    print("Training and validation results saved to results.json and learning curve saved to learning_curve.png")
