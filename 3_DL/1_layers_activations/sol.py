import torch
import torch.nn as nn


def function04(x: torch.Tensor, y: torch.Tensor):
    # weights = init_weights(x)
    layer = nn.Linear(in_features=x.shape[1], out_features=1, bias=True)
    # Training parameters
    learning_rate = 1e-2
    epochs = 5000  # Number of iterations for gradient descent
    
    # Gradient descent optimization
    for epoch in range(epochs):
        # Calculate predictions
        predictions = layer(x).ravel()
        
        # Calculate loss (mean squared error)
        loss = torch.mean((predictions - y) ** 2)

        print(f'MSE на шаге {epoch + 1} {loss.item():.5f}')
        
        # Compute gradients
        loss.backward()
        
        # Update weights using gradient descent
        with torch.no_grad():
            layer.weight -= learning_rate * layer.weight.grad
            layer.bias -= learning_rate * layer.bias.grad
        
        if loss < 0.3:
            return layer
        # Zero out the gradients to prevent accumulation
        layer.zero_grad()
    
    return layer


