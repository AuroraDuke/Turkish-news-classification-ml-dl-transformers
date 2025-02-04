import torch
import torch.nn as nn

class ComplexANN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim, cnn_filters, kernel_size, embedding_matrix=None):
        super(ComplexANN, self).__init__()
        # Embedding Layer
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # CNN Layer
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        
        # LSTM, BiLSTM, GRU
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, batch_first=True)
        self.bilstm = nn.LSTM(cnn_filters, hidden_dim, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(cnn_filters, hidden_dim, batch_first=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)  # Combine outputs from LSTM, BiLSTM, GRU
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # CNN
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, cnn_filters)
        
        # LSTM, BiLSTM, GRU
        lstm_out, _ = self.lstm(x)
        bilstm_out, _ = self.bilstm(x)
        gru_out, _ = self.gru(x)
        
        # Concatenate the outputs
        x = torch.cat((lstm_out[:, -1, :], bilstm_out[:, -1, :], gru_out[:, -1, :]), dim=1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)