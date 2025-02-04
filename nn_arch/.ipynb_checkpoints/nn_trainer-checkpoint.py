import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from nn_arch.simple_ann import SimpleANN  # ANN class from ann_model.py



def model_training(vectorization_method,model,criterion,optimizer,train_loader, val_loader ,num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    history = {"accuracy": [],"loss": [],"val_accuracy": [],"val_loss": []}
    
    y_test_cpu = [] # True labels
    predicted_cpu = []  # Predicted labels

     # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
             #Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
             # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
             # Backward pass and optimization
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                 # Move data to device
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # Accuracy calculation
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                if(epoch == num_epochs-1):
                    # Save final predictions
                    y_test_cpu.extend(predicted.cpu().numpy())
                    predicted_cpu.extend(batch_y.cpu().numpy())
                    
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        
         # Add history
        history["accuracy"].append(train_accuracy)
        history["loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
    model_name= f"{vectorization_method}_{model.__class__.__name__}"
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    return history , y_test_cpu, predicted_cpu