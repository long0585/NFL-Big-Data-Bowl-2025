import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, n_layers, d_ff, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_ff)  # Embedding layer for input features
        self.transformer = nn.Transformer(
            d_ff, n_heads, n_layers, n_layers, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_ff, output_dim)  # Output layer (binary classification)

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding layer
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x, x)  # Apply transformer
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.fc_out(x)  # Final prediction layer
        return x