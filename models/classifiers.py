import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCCClassifier(nn.Module):
    """
    Classifier for MFCC features
    """
    def __init__(self, num_classes, input_channels=13, dropout_rate=0.5):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of classes
            input_channels (int): Number of input channels (MFCC coefficients)
            dropout_rate (float): Dropout rate
        """
        super(MFCCClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions and pooling
        # Assuming input length is 1000 (can be adjusted based on actual input)
        self.fc_input_size = 256 * 125  # Adjust based on actual calculations
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor
            
        Returns:
            tensor: Output logits
        """
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, self.fc_input_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class HubertClassifier(nn.Module):
    """
    Classifier for HuBERT features
    """
    def __init__(self, num_classes, input_dim=768, hidden_dim=512, dropout_rate=0.5):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of classes
            input_dim (int): Input dimension (HuBERT embedding dimension)
            hidden_dim (int): Hidden layer dimension
            dropout_rate (float): Dropout rate
        """
        super(HubertClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor
            
        Returns:
            tensor: Output logits
        """
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # Concatenate forward and backward hidden states
        batch_size = hidden.shape[1]
        hidden = hidden.view(1, 2, batch_size, -1)  # (num_layers, num_directions, batch, hidden_size)
        forward_hidden = hidden[0, 0, :, :]  # (batch, hidden_size)
        backward_hidden = hidden[0, 1, :, :]  # (batch, hidden_size)
        concat_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size * 2)
        
        x = self.dropout(concat_hidden)
        x = self.fc(x)
        
        return x

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier
    """
    def __init__(self, num_classes, input_dim=768, num_heads=8, num_layers=4, dropout_rate=0.5):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of classes
            input_dim (int): Input dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate
        """
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor
            
        Returns:
            tensor: Output logits
        """
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch_size, input_dim)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x