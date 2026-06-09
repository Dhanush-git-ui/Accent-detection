import torch
import torch.nn as nn
import torch.nn.functional as F

class MFCCClassifier(nn.Module):
    """
    Enhanced CNN classifier for MFCC features.
    Supports 39-channel input (13 MFCC + 13 delta + 13 delta-delta).
    """
    def __init__(self, num_classes, input_channels=39, dropout_rate=0.3):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of classes
            input_channels (int): Number of input channels (default 39 for standard MFCC+delta+delta2)
            dropout_rate (float): Dropout rate
        """
        super(MFCCClassifier, self).__init__()
        
        # Enhanced CNN architecture
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)
        
        # Dynamically compute fc input size using a dummy pass (avoids hardcoding)
        # Input: (1, input_channels, 1000) -> after 4x MaxPool2: 1000//16 = 62 frames
        self._dummy_input_channels = input_channels
        self.fc_input_size = self._get_fc_input_size()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def _get_fc_input_size(self):
        """Compute flattened feature size via a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, self._dummy_input_channels, 1000)
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
            return x.view(1, -1).shape[1]
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor of shape (batch, channels, time)
            
        Returns:
            tensor: Output logits
        """
        # Support 2D input: (channels, time) -> add batch dim
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class HubertClassifier(nn.Module):
    """
    Enhanced Transformer classifier for HuBERT features with improved accuracy
    """
    def __init__(self, num_classes, input_dim=768, hidden_dim=512, num_heads=8, num_layers=4, dropout_rate=0.1):
        """
        Initialize classifier
        
        Args:
            num_classes (int): Number of classes
            input_dim (int): Input dimension (HuBERT embedding dimension)
            hidden_dim (int): Hidden layer dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate
        """
        super(HubertClassifier, self).__init__()
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, input_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for better training stability
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            tensor: Output logits
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Normalize input
        x = self.input_norm(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, dim, seq)
        x = self.pooling(x).squeeze(-1)  # (batch, dim)
        
        # Classification
        x = self.classifier(x)
        
        return x

class TransformerClassifier(nn.Module):
    """
    Enhanced Transformer-based classifier
    """
    def __init__(self, num_classes, input_dim=768, num_heads=8, num_layers=6, dropout_rate=0.3):
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
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, input_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 4, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (tensor): Input tensor
            
        Returns:
            tensor: Output logits
        """
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use [CLS] token representation (first token)
        cls_token = x[:, 0, :]
        
        # Classify
        x = self.classifier(cls_token)
        
        return x


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier combining MFCC and HuBERT models
    Weights HuBERT higher (60%) than MFCC (40%) for better accuracy
    """
    def __init__(self, mfcc_model, hubert_model, mfcc_weight=0.4, hubert_weight=0.6):
        """
        Initialize ensemble
        
        Args:
            mfcc_model: Trained MFCCClassifier
            hubert_model: Trained HubertClassifier
            mfcc_weight: Weight for MFCC predictions (default 0.4)
            hubert_weight: Weight for HuBERT predictions (default 0.6)
        """
        super(EnsembleClassifier, self).__init__()
        
        self.mfcc_model = mfcc_model
        self.hubert_model = hubert_model
        self.mfcc_weight = mfcc_weight
        self.hubert_weight = hubert_weight
        
        # Freeze both models
        for param in self.mfcc_model.parameters():
            param.requires_grad = False
        for param in self.hubert_model.parameters():
            param.requires_grad = False
    
    def forward(self, mfcc_input, hubert_input):
        """
        Forward pass
        
        Args:
            mfcc_input: MFCC features (batch, 39, time)
            hubert_input: HuBERT features (batch, time, 768)
            
        Returns:
            tensor: Ensemble logits
        """
        # Get predictions from both models
        mfcc_logits = self.mfcc_model(mfcc_input)
        hubert_logits = self.hubert_model(hubert_input)
        
        # Convert to probabilities
        mfcc_probs = F.softmax(mfcc_logits, dim=1)
        hubert_probs = F.softmax(hubert_logits, dim=1)
        
        # Weighted ensemble
        ensemble_probs = self.mfcc_weight * mfcc_probs + self.hubert_weight * hubert_probs
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-10)
        
        return ensemble_logits