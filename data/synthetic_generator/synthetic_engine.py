import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image, ImageDraw

class ConditionalGANGenerator(nn.Module):
    """
    Conditional Generator for 5-modality cognitive metrics & latent features.
    Inputs: random noise vector z (dim 64) + target severity labels y (dim 4).
    Outputs: concatenated feature vector covering tabular & embedding representations across modalities.
    """
    def __init__(self, noise_dim=64, label_dim=4, output_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.net(x)

class ConditionalGANDiscriminator(nn.Module):
    """
    Discriminator ensuring generated multimodal feature embeddings match realistic distributions.
    """
    def __init__(self, input_dim=128, label_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)

class SyntheticCognitiveGenerator:
    """
    High-level generator API for physics-informed & GAN-augmented multimodal sample creation.
    Synthesizes aligned data across Gaze, Handwriting, Speech, Drawing, and EEG.
    """
    def __init__(self, noise_dim=64, label_dim=4):
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.generator = ConditionalGANGenerator(noise_dim, label_dim, output_dim=128)
        self.discriminator = ConditionalGANDiscriminator(input_dim=128, label_dim=4)
        self.is_trained = False

    def train_cgan(self, epochs=50, batch_size=32, lr=0.0002):
        """Train conditional GAN on simulated realistic baseline seeds."""
        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            # Create synthetic realistic "real" seeds
            labels = torch.rand(batch_size, self.label_dim)
            real_features = torch.randn(batch_size, 128) * 0.5 + (labels.mean(dim=1, keepdim=True) * 0.8)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_targets = torch.ones(batch_size, 1)
            fake_targets = torch.zeros(batch_size, 1)
            
            d_real_loss = criterion(self.discriminator(real_features, labels), real_targets)
            
            z = torch.randn(batch_size, self.noise_dim)
            fake_features = self.generator(z, labels)
            d_fake_loss = criterion(self.discriminator(fake_features.detach(), labels), fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            g_loss = criterion(self.discriminator(fake_features, labels), real_targets)
            g_loss.backward()
            g_optimizer.step()
            
        self.is_trained = True

    def generate_multimodal_sample(self, dyslexia_sev=0.0, dysgraphia_sev=0.0, dyscalculia_sev=0.0, adhd_sev=0.0):
        """
        Generate a synchronized set of 5 modality signals conditioned on target severity scores.
        Returns tabular metrics and procedurally rendered modality images.
        """
        labels = np.array([dyslexia_sev, dysgraphia_sev, dyscalculia_sev, adhd_sev], dtype=np.float32)
        
        # 1. Gaze Features & Scanpath Rendering
        fixation_duration = 200 + dyslexia_sev * 250 + np.random.normal(0, 20)
        regression_count = int(2 + dyslexia_sev * 12 + np.random.poisson(1))
        saccade_velocity = 300 - dyslexia_sev * 120 + np.random.normal(0, 15)
        gaze_tabular = np.array([fixation_duration, regression_count, saccade_velocity, 
                                 dyslexia_sev, adhd_sev, np.random.rand()], dtype=np.float32)
        
        # Render scanpath image (128x128 RGB)
        gaze_img = Image.new('RGB', (128, 128), color=(240, 240, 245))
        draw = ImageDraw.Draw(gaze_img)
        points = []
        num_fixations = int(5 + dyslexia_sev * 15)
        cx, cy = 20, 30
        for _ in range(num_fixations):
            cx += np.random.randint(5, 25)
            cy += np.random.randint(-5, 5)
            if cx > 120:
                cx = 20
                cy += 20
            points.append((cx, cy))
            r = int(3 + dyslexia_sev * 6)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(230, 50, 50, 150))
        if len(points) > 1:
            draw.line(points, fill=(50, 50, 200), width=1)
            
        # 2. Handwriting Time-Series Stylus Dynamics (50 timesteps x 4 features: dx, dy, pressure, tremor)
        ts_steps = 50
        tremor_freq = 0.2 if dysgraphia_sev > 0.3 else 0.05
        t = np.linspace(0, 4*np.pi, ts_steps)
        dx = np.cos(t) + np.random.normal(0, tremor_freq, ts_steps)
        dy = np.sin(t) + np.random.normal(0, tremor_freq, ts_steps)
        pressure = np.clip(0.6 - dysgraphia_sev*0.4 + np.random.normal(0, 0.1, ts_steps), 0.1, 1.0)
        tremor = np.abs(np.diff(dx, prepend=dx[0])) * (1.0 + dysgraphia_sev * 2.0)
        handwriting_ts = np.stack([dx, dy, pressure, tremor], axis=1).astype(np.float32)
        
        # 3. Speech Prosody Tabular Vector (8 features)
        pause_ratio = 0.1 + dyslexia_sev * 0.35 + adhd_sev * 0.1
        pitch_variance = 40 + adhd_sev * 35 + np.random.normal(0, 5)
        phoneme_error_rate = 0.05 + dyslexia_sev * 0.4
        speaking_rate_wpm = 140 - dyslexia_sev * 60 - adhd_sev * 20
        speech_tabular = np.array([pause_ratio, pitch_variance, phoneme_error_rate, speaking_rate_wpm,
                                  dyslexia_sev, adhd_sev, np.random.rand(), np.random.rand()], dtype=np.float32)
                                  
        # 4. Drawing Test Rendering (Clock & Bender-Gestalt Distortion)
        drawing_img = Image.new('RGB', (128, 128), color=(255, 255, 255))
        draw_d = ImageDraw.Draw(drawing_img)
        # Clock face circle
        draw_d.ellipse([14, 14, 114, 114], outline=(20, 20, 20), width=2)
        # Distortion based on dyscalculia & dysgraphia
        center_x, center_y = 64, 64
        distortion = dyscalculia_sev * 15 + dysgraphia_sev * 10
        # Draw clock hands with spatial distortion
        hand1_end = (center_x + 30 + np.random.uniform(-distortion, distortion),
                     center_y - 20 + np.random.uniform(-distortion, distortion))
        hand2_end = (center_x - 15 + np.random.uniform(-distortion, distortion),
                     center_y + 35 + np.random.uniform(-distortion, distortion))
        draw_d.line([(center_x, center_y), hand1_end], fill=(10, 10, 10), width=3)
        draw_d.line([(center_x, center_y), hand2_end], fill=(10, 10, 10), width=2)
        
        # 5. EEG Spectral Power Tabular Vector (5 features: delta, theta, alpha, beta, theta_beta_ratio)
        theta_power = 12.0 + adhd_sev * 15.0 + np.random.normal(0, 1.5)
        beta_power = 18.0 - adhd_sev * 8.0 + np.random.normal(0, 1.0)
        alpha_power = 10.0 + np.random.normal(0, 1.0)
        delta_power = 5.0 + np.random.normal(0, 0.5)
        theta_beta_ratio = theta_power / max(beta_power, 1e-4)
        eeg_tabular = np.array([delta_power, theta_power, alpha_power, beta_power, theta_beta_ratio], dtype=np.float32)
        
        return {
            'gaze_tabular': gaze_tabular,
            'gaze_img': gaze_img,
            'handwriting_ts': handwriting_ts,
            'speech_tabular': speech_tabular,
            'drawing_img': drawing_img,
            'eeg_tabular': eeg_tabular,
            'labels': labels
        }
