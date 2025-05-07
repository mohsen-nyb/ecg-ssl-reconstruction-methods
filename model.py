import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ECG_CNN_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, kernel_size=15, dropout=0.1, alpha=0.1, seed=42):
        super(ECG_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.isvae = False
        
        self.signal_length = signal_length
        padding_size = int((kernel_size - 1) / 2)
            
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2) 
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)

        self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)  

        self.alpha = alpha
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout*2)


    def _get_flatten_size(self):
        """Get the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, self.signal_length)
            x = F.leaky_relu(self.conv1(dummy_input))
            x = F.leaky_relu(self.conv2(x))
            x = self.avgpool1(x)  
            x = F.leaky_relu(self.conv3(x))
            x = self.avgpool2(x)  
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.avgpool1(x)
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.avgpool2(x)
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        
        x = torch.flatten(x, 1)
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=self.alpha))
        
        return x



class ECG_CNN_VAE_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, kernel_size=15, dropout=0.1, alpha=0.1, seed=42):
        super(ECG_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.isvae = True
        self.signal_length = signal_length
        padding_size = int((kernel_size - 1) / 2)
            
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2) 
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)

        self.flatten_size = self._get_flatten_size() 
        self.fc_mu = nn.Linear(self.flatten_size, embedded_size)
        self.fc_logvar = nn.Linear(self.flatten_size, embedded_size)

        self.alpha = alpha
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout*2)


    def _get_flatten_size(self):
        """Get the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, self.signal_length)
            x = F.leaky_relu(self.conv1(dummy_input))
            x = F.leaky_relu(self.conv2(x))
            x = self.avgpool1(x)  
            x = F.leaky_relu(self.conv3(x))
            x = self.avgpool2(x)  
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.avgpool1(x)
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.avgpool2(x)
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar






class ECG_Decoder(nn.Module):
    def __init__(self, encoded_size=256, output_channels=12, target_length=5000, seed=42):
        super(ECG_Decoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.target_length = target_length

        # This will map from the 256-dim latent vector to a lower-resolution feature map
        self.fc = nn.Linear(encoded_size, 64 * 40)  # Initialize to (B, 64, 40)

        # Transposed convolutions to upsample progressively
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=15, stride=2, padding=7, output_padding=1),  # ~40 -> ~80
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(64, 64, kernel_size=15, stride=2, padding=7, output_padding=1),  # ~80 -> ~160
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),  # ~160 -> ~320
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),  # ~320 -> ~640
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(16, output_channels, kernel_size=15, stride=2, padding=7, output_padding=1),  # ~640 -> ~1280
            nn.Tanh()
        )

        # Final interpolation layer to ensure output length matches exactly 5000
        self.upsample_to_target = nn.Upsample(size=target_length, mode='linear', align_corners=True)

    def forward(self, x):
        x = self.fc(x)  # B x (64*40)
        x = x.view(x.size(0), 64, 40)  # Reshape to (B, 64, 40)
        x = self.deconv_layers(x)     # Now shape is (B, 12, ~1280)
        x = self.upsample_to_target(x)  # Interpolate to (B, 12, 5000)
        return x


class ECG_Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ECG_Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class ECG_VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ECG_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class ECG_Classifier(nn.Module):
    def __init__(self, encoder, embedded_size=256, num_classes=1, dropout=0.1):
        super(ECG_Classifier, self).__init__()
        self.encoder = encoder  # Use pretrained encoder
        
        # Two-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedded_size, num_classes),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        if self.encoder.isvae:
            x, _ = self.encoder(x)  # Get encoded features
        else:
            x = self.encoder(x)  # Get encoded features
        out = self.classifier(x)  # Classifier prediction
        return out



    



# # Instantiate the model components with reduced batch size
# encoder = ECG_Encoder()
# decoder = ECG_Decoder(encoded_size=256, output_channels=12, target_length=5000)
# autoencoder = ECG_Autoencoder(encoder, decoder)

# # Create dummy input with shape (B=2, 12, 5000)
# dummy_input = torch.randn(2, 12, 5000)

# # Run without gradient tracking to save memory
# with torch.no_grad():
#     reconstructed = autoencoder(dummy_input)

# # Output the shape of the reconstructed signal
# reconstructed.shape
