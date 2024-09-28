import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
		self.conv1_BN = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv2_BN = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
		self.conv3_BN = nn.BatchNorm2d(32)
		# MLP encoder
		self.fc1 = nn.Linear(512, 256)
		self.fc1_BN = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 128)
		self.fc2_BN = nn.BatchNorm1d(128)
		self.latent_dim = args.latent_dim
		self.z_out = nn.Linear(128, self.latent_dim)
		# Nonlinearities
		self.leaky_relu = nn.LeakyReLU()
	def forward(self, x, device):
		# Convolutional encoder
		conv1_out = self.leaky_relu(self.conv1_BN(self.conv1(x)))
		conv2_out = self.leaky_relu(self.conv2_BN(self.conv2(conv1_out)))
		conv3_out = self.leaky_relu(self.conv3_BN(self.conv3(conv2_out)))
		# Flatten output of conv. net
		conv3_out_flat = torch.flatten(conv3_out, 1)
		# MLP encoder
		fc1_out = self.leaky_relu(self.fc1_BN(self.fc1(conv3_out_flat)))
		fc2_out = self.leaky_relu(self.fc2_BN(self.fc2(fc1_out)))
		z = self.z_out(fc2_out)
		return z, conv3_out_flat

class Class_out(nn.Module):
	def __init__(self, args):
		super(Class_out, self).__init__()
		# Feedforward layer
		self.fc = nn.Linear(args.latent_dim, 10)
	def forward(self, z):
		y = self.fc(z)
		return y

class Conf_out(nn.Module):
	def __init__(self, args):
		super(Conf_out, self).__init__()
		# Feedforward layer
		self.fc = nn.Linear(args.latent_dim, 1)
		# Nonlinearities
		self.sigmoid = nn.Sigmoid()
	def forward(self, z):
		conf = self.sigmoid(self.fc(z))
		return conf
	
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

noise_range = [0.1, 1.0]
signal_range = [1.0, 2.0]

def CNN_denoise(encoder, classifier, conf_out, train_loader, val_loader, test_loader, criterion_class, criterion_conf, optimizer, device):
    encoder.train()
    classifier.train()
    conf_out.train()

    best_loss = float('inf')  # Initialize best loss as infinity
    best_model = None         # To store the best model parameters
    best_acc = 0.0            # Best accuracy for the model with min loss
    best_conf = 0.0           # Best confidence for the model with min loss

    # Training loop
    for batch_images, batch_labels in train_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Scale signal
        signal = ((torch.rand(batch_images.shape[0]) * (signal_range[1] - signal_range[0])) + signal_range[0]).to(device)
        batch_images = batch_images * signal.view(-1, 1, 1, 1)
        
        # Scale to [-1, 1]
        batch_images = (batch_images - 0.5) / 0.5
        
        # Add noise
        noise = (torch.rand(batch_images.shape[0]) * (noise_range[1] - noise_range[0])) + noise_range[0]
        noise = noise.view(-1, 1, 1, 1).repeat(1, 1, batch_images.shape[2], batch_images.shape[3])
        noise = (torch.randn(batch_images.shape) * noise).to(device)
        batch_images = batch_images + noise
        
        # Threshold image
        batch_images = nn.Hardtanh()(batch_images)

        # Forward pass
        z, conv_flat = encoder(batch_images, device)
        class_preds = classifier(z)
        conf_preds = conf_out(z)
        avg_conf = conf_preds.mean().item()

        # Loss calculation
        class_loss = criterion_class(class_preds, batch_labels)
        conf_loss = criterion_conf(conf_preds.squeeze(), (batch_labels == torch.argmax(class_preds, dim=1)).float())
        loss = class_loss + conf_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        _, predicted = torch.max(class_preds, 1)
        correct = (predicted == batch_labels).sum().item()
        total = batch_labels.size(0)
        acc = correct / total

        # Check if current model is the best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_acc = acc
            best_conf = avg_conf
            best_model = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'conf_out': conf_out.state_dict()
            }
    
    # Validation part
    encoder.eval()
    classifier.eval()
    conf_out.eval()
    
    best_z = []
    best_conv_flat = []
    accs = []
    confs = []
    
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            # Scale signal
            batch_images = batch_images * 1  # No signal during validation
            # Scale to [-1, 1]
            batch_images = (batch_images - 0.5) / 0.5
            # Add noise
            batch_images = batch_images + (torch.randn(batch_images.shape) * 0.0).to(device)  # No noise during validation
            # Threshold image
            batch_images = nn.Hardtanh()(batch_images)

            # Forward pass through the best model
            z, conv_flat = encoder(batch_images, device)
            class_preds = classifier(z)
            conf_preds = conf_out(z)
            avg_conf = conf_preds.mean().item()

            # Accuracy calculation
            _, predicted = torch.max(class_preds, 1)
            correct = (predicted == batch_labels).sum().item()
            total = batch_labels.size(0)
            acc = correct / total
            
            accs.append(acc)
            confs.append(avg_conf)

            # Collect latent vectors
            best_z.append(z.detach().cpu())
            best_conv_flat.append(conv_flat.detach().cpu())

    # Final train set evaluation
    train_acc = np.mean(accs)
    train_conf = np.mean(confs)
    
    best_z = torch.cat(best_z, dim=0)
    best_conv_flat = torch.cat(best_conv_flat, dim=0)

    # Test set evaluation
    test_z = []
    test_conv_flat = []
    test_accs = []
    test_confs = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            # Scale signal
            batch_images = batch_images * 1  # No signal during testing
            # Scale to [-1, 1]
            batch_images = (batch_images - 0.5) / 0.5
            # Add noise (if desired, currently 0.0)
            batch_images = batch_images + (torch.randn(batch_images.shape) * 0.0).to(device)  # No noise during testing
            # Threshold image
            batch_images = nn.Hardtanh()(batch_images)

            # Forward pass
            z, conv_flat = encoder(batch_images, device)
            class_preds = classifier(z)
            conf_preds = conf_out(z)
            avg_conf = conf_preds.mean().item()

            # Accuracy calculation
            _, predicted = torch.max(class_preds, 1)
            correct = (predicted == batch_labels).sum().item()
            total = batch_labels.size(0)
            acc = correct / total

            test_accs.append(acc)
            test_confs.append(avg_conf)

            # Collect latent vectors
            test_z.append(z.detach().cpu())
            test_conv_flat.append(conv_flat.detach().cpu())

    test_acc = np.mean(test_accs)
    test_conf = np.mean(test_confs)

    test_z = torch.cat(test_z, dim=0)
    test_conv_flat = torch.cat(test_conv_flat, dim=0)

    # Collect stats
    stats = {
        'train_acc': np.round(train_acc, 4),
        'train_conf': np.round(train_conf, 4),
        'test_acc': np.round(test_acc, 4),
        'test_conf': np.round(test_conf, 4),
        'best_loss': np.round(best_loss, 4),
        'best_acc': np.round(best_acc, 4),
        'best_conf': np.round(best_conf, 4)
    }

    return best_model, best_z, best_conv_flat, test_z, test_conv_flat, stats


def PCA_reduction(train_images, test_images, latent_dim, random_seed):
    # Flatten train and test images for PCA (batch_size, 1, 28, 28) -> (batch_size, 28*28)
    train_images_flat = train_images.reshape(train_images.shape[0], 28*28)
    test_images_flat = test_images.reshape(test_images.shape[0], 28*28)

    # Apply PCA to reduce dimensionality to latent_dim
    pca = PCA(n_components=latent_dim, random_state=random_seed)
    train_pca = pca.fit_transform(train_images_flat)  # Fit on training data
	# predict on test data
    test_pca = pca.transform(test_images_flat)  # Transform test data
	
	# use inverse_transform to encode the images
    train_reconstructed = pca.inverse_transform(train_pca)
    test_reconstructed = pca.inverse_transform(pca.transform(test_images_flat))  # Transform test data
	
    return train_pca, test_pca, train_reconstructed, test_reconstructed