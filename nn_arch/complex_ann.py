import torch
import torch.nn as nn
import numpy as np

class ComplexANN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, input_length=100, hidden_dim=64, cnn_filters=128, kernel_size=3, pretrained_embedding=None):
        super(ComplexANN, self).__init__()
        
        # Embedding layer
        if pretrained_embedding is not None:
            # Use pre-trained embedding
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embedding, dtype=torch.float32), freeze=False)
        else:
            # Initialize randomly if no pre-trained embedding is provided
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu_cnn = nn.ReLU()
        self.batch_norm_cnn = nn.BatchNorm1d(cnn_filters)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(cnn_filters, hidden_dim, bidirectional=True, batch_first=True)
        
        self.dropout1 = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 2)  # Bidirectional LSTM doubles the output size
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2 * (input_length // 2), hidden_dim)  # Adjust input size to match LSTM output
        self.relu = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)  # Embedding layer
        x = x.permute(0, 2, 1)  # Swap dimensions for Conv1d (batch, channels, seq_len)
        x = self.conv1(x)  # Convolutional layer
        x = self.relu_cnn(x)
        x = self.batch_norm_cnn(x)
        x = self.max_pool(x)  # Max pooling
        x = x.permute(0, 2, 1)  # Swap dimensions back for LSTM (batch, seq_len, channels)
        
        x, _ = self.bilstm(x)  # Bidirectional LSTM layer
        
        x = self.dropout1(x)  # Dropout after LSTM
        x = x.permute(0, 2, 1)  # Swap dimensions for BatchNorm1d (batch, channels, seq_len)
        x = self.batch_norm1(x)
        x = x.permute(0, 2, 1)  # Swap dimensions back

        x = self.flatten(x)  # Flatten the tensor for fully connected layers
        
        x = self.fc1(x)  # First fully connected layer
        x = self.relu(x)
        x = self.batch_norm2(x)  # Batch normalization
        x = self.dropout2(x)  # Dropout
        
        x = self.fc2(x)  # Final fully connected layer
        x = self.softmax(x)  # Softmax activation for output
        
        return x