#!/usr/bin/env python3
"""
Deep Value Network Training Script

This script trains a deep neural network to predict the expected value of poker game states
for use in the DeepStack engine (RLCard Superhuman Protocol - Pillar 3).

The network takes a game state vector and predicts its expected value (single float).
Designed for cloud training in Google Colab with robust checkpointing to Google Drive.

Usage (in Google Colab):
    python training/train_value_network.py --output-dir /content/drive/MyDrive/poker-ai/models \\
                                          --epochs 100 \\
                                          --batch-size 32
"""

import argparse
import logging
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime

# RLCard for game state generation
import rlcard
from rlcard.agents import CFRAgent, RandomAgent


def setup_logging(output_dir: str):
    """Configure logging for the training process."""
    log_file = Path(output_dir) / f"value_network_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DeepValueNetwork(nn.Module):
    """
    Deep neural network for poker game state value prediction.
    
    Architecture:
    - Input: Game state vector (variable size, padded/truncated to fixed size)
    - Hidden layers: Fully connected with ReLU activation and dropout
    - Output: Single value prediction (expected payoff)
    """
    
    def __init__(self, input_size: int = 512, hidden_sizes: List[int] = [1024, 512, 256]):
        super(DeepValueNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build the network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer (no activation - raw value prediction)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x).squeeze(-1)  # Remove last dimension for scalar output


def create_game_state_vector(state: Dict[str, Any], max_size: int = 512) -> np.ndarray:
    """
    Convert RLCard game state to fixed-size numerical vector.
    
    This is a simplified encoding - in a real implementation, this would be
    much more sophisticated with proper feature engineering.
    """
    try:
        # Extract key features from the state
        features = []
        
        # Legal actions (one-hot encoded)
        legal_actions = state.get('legal_actions', {})
        action_vector = np.zeros(10)  # Assume max 10 possible actions
        for i, action in enumerate(legal_actions.keys()):
            if i < 10:
                action_vector[i] = 1
        features.extend(action_vector)
        
        # Observation features (simplified)
        obs = state.get('obs', [])
        if isinstance(obs, (list, np.ndarray)):
            obs_vector = np.array(obs).flatten()
            # Truncate or pad to standard size
            if len(obs_vector) > 400:
                obs_vector = obs_vector[:400]
            elif len(obs_vector) < 400:
                padded = np.zeros(400)
                padded[:len(obs_vector)] = obs_vector
                obs_vector = padded
            features.extend(obs_vector)
        else:
            features.extend(np.zeros(400))
        
        # Convert to numpy array and ensure fixed size
        feature_vector = np.array(features, dtype=np.float32)
        
        if len(feature_vector) > max_size:
            feature_vector = feature_vector[:max_size]
        elif len(feature_vector) < max_size:
            padded = np.zeros(max_size, dtype=np.float32)
            padded[:len(feature_vector)] = feature_vector
            feature_vector = padded
            
        return feature_vector
        
    except Exception as e:
        logging.warning(f"Error creating state vector: {e}")
        return np.zeros(max_size, dtype=np.float32)


def generate_training_data(num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data by simulating poker hands and recording state-value pairs.
    
    Returns:
        X: Game state vectors (num_samples, input_size)
        y: Expected values (num_samples,)
    """
    logging.info(f"Generating {num_samples} training samples...")
    
    # Create environment and agents
    env = rlcard.make('no-limit-holdem', config={'seed': 42})
    cfr_agent = CFRAgent(env)
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    X = []
    y = []
    
    for sample_id in range(num_samples):
        try:
            # Reset environment
            state, player_id = env.reset()
            game_states = []
            
            # Play out the hand and collect states
            while not env.is_over():
                # Record current state
                state_vector = create_game_state_vector(state)
                game_states.append((state_vector, player_id))
                
                # Choose action based on player
                if player_id == 0:
                    action, _ = cfr_agent.eval_step(state)
                else:
                    action = random_agent.eval_step(state)[0]
                
                state, player_id = env.step(action)
            
            # Get final payoffs
            payoffs = env.get_payoffs()
            
            # Assign values to states based on final outcomes
            for state_vector, p_id in game_states:
                expected_value = payoffs[p_id]
                X.append(state_vector)
                y.append(expected_value)
        
        except Exception as e:
            logging.warning(f"Error in sample {sample_id}: {e}")
            continue
            
        # Progress reporting
        if (sample_id + 1) % 1000 == 0:
            logging.info(f"Generated {sample_id + 1}/{num_samples} samples")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    logging.info(f"Training data generated: X shape {X.shape}, y shape {y.shape}")
    return X, y


def create_data_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                       val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Split into training and validation
    split_idx = int(len(X) * (1 - val_split))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Data loaders created: Train {len(train_loader)} batches, Val {len(val_loader)} batches")
    return train_loader, val_loader


def train_epoch(model: DeepValueNetwork, train_loader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model: DeepValueNetwork, val_loader: DataLoader, 
                  criterion: nn.Module, device: torch.device) -> float:
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model: DeepValueNetwork, optimizer: optim.Optimizer, 
                   epoch: int, train_loss: float, val_loss: float, 
                   output_dir: str) -> str:
    """Save training checkpoint."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"value_network_epoch_{epoch}.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': {
            'input_size': model.input_size,
            'hidden_sizes': model.hidden_sizes
        }
    }, checkpoint_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    return str(checkpoint_path)


def export_to_onnx(model: DeepValueNetwork, output_path: str, input_size: int = 512):
    """Export the trained model to ONNX format."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['game_state'],
        output_names=['expected_value'],
        dynamic_axes={
            'game_state': {0: 'batch_size'},
            'expected_value': {0: 'batch_size'}
        }
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logging.info(f"Model exported to ONNX: {output_path}")


def train_value_network(output_dir: str, epochs: int = 100, batch_size: int = 32,
                       learning_rate: float = 0.001, num_samples: int = 10000) -> str:
    """
    Main training function for the Deep Value Network.
    
    Returns:
        Path to the final ONNX model
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Generate training data
    X, y = generate_training_data(num_samples)
    train_loader, val_loader = create_data_loaders(X, y, batch_size)
    
    # Create model
    input_size = X.shape[1]
    model = DeepValueNetwork(input_size=input_size).to(device)
    
    # Training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    logging.info(f"Starting training: {epochs} epochs, batch size {batch_size}")
    
    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:3d}/{epochs}: Train Loss {train_loss:.4f}, "
                    f"Val Loss {val_loss:.4f}, LR {current_lr:.2e}")
        
        # Save checkpoint
        if epoch % 10 == 0 or val_loss < best_val_loss:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, train_loss, val_loss, output_dir)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_path
                logging.info(f"New best model saved (val_loss: {val_loss:.4f})")
    
    # Export best model to ONNX
    if best_model_path:
        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Export to ONNX
        onnx_path = Path(output_dir) / "deep_value_network_v1.onnx"
        export_to_onnx(model, str(onnx_path), input_size)
        
        # Save training metadata
        metadata = {
            'training_completed': datetime.now().isoformat(),
            'epochs_trained': epochs,
            'best_val_loss': best_val_loss,
            'input_size': input_size,
            'model_architecture': {
                'hidden_sizes': model.hidden_sizes,
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_training_samples': num_samples
            }
        }
        
        metadata_path = Path(output_dir) / "deep_value_network_v1_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Training completed! Final model: {onnx_path}")
        return str(onnx_path)
    
    else:
        raise RuntimeError("No best model was saved during training")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train Deep Value Network for poker state evaluation')
    parser.add_argument('--output-dir', required=True,
                       help='Directory to save the trained model (Google Drive path recommended)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of training samples to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir))
    
    logger.info("🧠 Deep Value Network Training")
    logger.info("=" * 40)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Training samples: {args.num_samples}")
    
    try:
        # Train the model
        final_model_path = train_value_network(
            str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_samples=args.num_samples
        )
        
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"   Final model: {final_model_path}")
        logger.info(f"   Download this file to your local models/ directory")
        logger.info(f"   for integration with the Synthesizer module")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())