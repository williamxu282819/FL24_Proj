import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Balanced
# noise_range = [0.1, 1.0]
# signal_range = [1.0, 2.0]

# Noisless
noise_range = [0.1, 0.2]
signal_range = [2.0, 4.0]

# Noisy
noise_range = [0.9, 1.0]
signal_range = [0.5, 1.0]

def CNN_denoise(encoder, classifier, conf_out, train_loader, test_loader, criterion_class, criterion_conf, optimizer, device):
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

    # Test set evaluation
    test_z = []
    test_accs = []
    test_confs = []
    predicted_labels = []
    true_labels = []

    all_class_preds = []
    all_conf_preds = []

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

            # save the predictions
            all_class_preds.append(class_preds.detach().cpu())
            all_conf_preds.append(conf_preds.detach().cpu())

            # Accuracy calculation
            _, predicted = torch.max(class_preds, 1)
            correct = (predicted == batch_labels).sum().item()
            total = batch_labels.size(0)
            acc = correct / total

            test_accs.append(acc)
            test_confs.append(avg_conf)

            # Collect latent vectors
            test_z.append(z.detach().cpu())

            # Collect predicted labels
            predicted_labels.append(predicted)
            true_labels.append(batch_labels)

            
    test_acc = np.mean(test_accs)
    test_conf = np.mean(test_confs)
	
    test_z = torch.cat(test_z, dim=0)
    
    # Collect stats
    stats = {
        'test_acc': np.round(test_acc, 4),
        'test_conf': np.round(test_conf, 4),
        'best_loss': np.round(best_loss, 4),
        'best_acc': np.round(best_acc, 4),
        'best_conf': np.round(best_conf, 4)
    }

    return best_model, test_z, stats, predicted_labels, true_labels, all_class_preds, all_conf_preds


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