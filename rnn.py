import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# -----------------------------
# 1. Choose GPU if available
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. RNN Model Definition
# -----------------------------
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        
        # RNN layer: (seq_len, batch_size, input_dim) => (seq_len, batch_size, hidden_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, self.num_layers, nonlinearity='tanh', batch_first=False)
        
        # Fully connected layer => 5 sentiment classes
        self.fc = nn.Linear(hidden_dim, 5)
        
        # LogSoftmax for classification
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Negative log likelihood loss
        self.loss_function = nn.NLLLoss()

    def compute_loss(self, predicted_output, true_label):
        # predicted_output: shape (batch_size, 5)
        # true_label: shape (batch_size)
        return self.loss_function(predicted_output, true_label)

    def forward(self, inputs):
        """
        inputs shape: (seq_len, batch_size, input_dim)
        We create the hidden state on the same device as inputs.
        """
        batch_size = inputs.size(1)
        
        # Initialize hidden state on correct device
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=inputs.device)
        
        # Run RNN
        rnn_output, hidden_state = self.rnn(inputs, hidden_state)
        
        # Take the last hidden state => shape (batch_size, hidden_dim)
        last_hidden = hidden_state[-1]
        
        # Pass through fully connected layer => shape (batch_size, 5)
        output = self.fc(last_hidden)
        
        # Convert to log probabilities => shape (batch_size, 5)
        predicted_output = self.softmax(output)
        return predicted_output

# -----------------------------
# 3. Load Training & Validation Data
# -----------------------------
def load_data(train_path, val_path):
    with open(train_path) as train_file:
        train_raw = json.load(train_file)
    with open(val_path) as val_file:
        val_raw = json.load(val_file)

    # Each entry: {"text": "some words here", "stars": 1..5}
    # We'll store them as (list_of_words, label_index)
    # label_index = stars - 1  =>  0..4
    train_data = [(entry["text"].split(), int(entry["stars"]) - 1) for entry in train_raw]
    val_data   = [(entry["text"].split(), int(entry["stars"]) - 1) for entry in val_raw]
    return train_data, val_data

# -----------------------------
# 4. Plot Learning Curve
# -----------------------------
def plot_learning_curve(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Validation Accuracy over Epochs (RNN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_rnn.png')
    plt.show()

# -----------------------------
# 5. Main: Train & Validate
# -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, required=True, help="Hidden dimension size")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--train_data', required=True, help="Path to training data")
    parser.add_argument('--val_data', required=True, help="Path to validation data")
    args = parser.parse_args()

    print("Loading data...")
    train_data, val_data = load_data(args.train_data, args.val_data)

    # For simplicity, let's assume input_dim=50
    input_dim = 50

    print("Initializing model...")
    model = RNN(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # We will store training losses and validation accuracies for plotting
    train_losses = []
    val_accuracies = []

    # Number of epochs
    for epoch in range(args.epochs):
        # Shuffle training data each epoch
        random.shuffle(train_data)
        model.train()
        
        correct, total = 0, 0
        loss_total, loss_count = 0.0, 0
        
        print(f"\nTraining epoch {epoch+1} / {args.epochs}")

        # Let's define a mini-batch size of 32
        batch_size = 32
        for batch_start in tqdm(range(0, len(train_data), batch_size)):
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for i in range(batch_size):
                if batch_start + i >= len(train_data):
                    break

                # 1) Get data from train_data
                input_words, label = train_data[batch_start + i]
                
                # 2) Create random embeddings for each word => shape (seq_len, input_dim)
                input_vectors = np.random.rand(len(input_words), input_dim)

                # 3) Convert to tensor => shape (seq_len, 1, input_dim)
                input_tensor = torch.tensor(input_vectors, dtype=torch.float32).unsqueeze(1).to(device)
                
                # 4) Forward pass
                output = model(input_tensor)
                
                # 5) Predictions
                predicted_label = torch.argmax(output, dim=1)
                correct += int(predicted_label.item() == label)
                total += 1

                # 6) Compute loss
                label_tensor = torch.tensor([label], dtype=torch.long, device=device)
                batch_loss += model.compute_loss(output, label_tensor)

            # 7) Backprop
            batch_loss.backward()
            optimizer.step()

            loss_total += batch_loss.item()
            loss_count += 1
        
        # Compute average training loss
        train_loss = loss_total / loss_count if loss_count > 0 else 0
        train_accuracy = correct / total if total > 0 else 0
        
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}: Training accuracy = {train_accuracy:.4f}, Training loss = {train_loss:.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for input_words, label in val_data:
                # Create random embeddings => shape (seq_len, input_dim)
                input_vectors = np.random.rand(len(input_words), input_dim)

                # Convert to tensor => shape (seq_len, 1, input_dim)
                input_tensor = torch.tensor(input_vectors, dtype=torch.float32).unsqueeze(1).to(device)
                
                # Forward pass
                output = model(input_tensor)
                
                # Prediction
                predicted_label = torch.argmax(output, dim=1)
                correct += int(predicted_label.item() == label)
                total += 1
        
        val_accuracy = correct / total if total > 0 else 0
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}: Validation accuracy = {val_accuracy:.4f}")

    # ---------------------------
    # Save Results
    # ---------------------------
    import json
    results = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies
    }
    with open("results_rnn.json", "w") as f:
        json.dump(results, f)

    # ---------------------------
    # Plot Learning Curve
    # ---------------------------
    epochs_range = list(range(1, args.epochs + 1))
    plot_learning_curve(epochs_range, train_losses, val_accuracies)
    print("\nTraining complete! Results saved to results_rnn.json and learning_curve_rnn.png.")
