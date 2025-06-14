"""Shared pytest fixtures and configuration for all tests."""
import os
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_tensor():
    """Create a sample PyTorch tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "scale": 4,
            "num_channels": 3,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
        },
        "dataset": {
            "train_path": "/tmp/train",
            "val_path": "/tmp/val",
        }
    }


@pytest.fixture
def mock_model_weights(temp_dir):
    """Create mock model weights file."""
    weights_path = temp_dir / "model_weights.pth"
    mock_state_dict = {
        "conv1.weight": torch.randn(64, 3, 3, 3),
        "conv1.bias": torch.randn(64),
    }
    torch.save(mock_state_dict, weights_path)
    return weights_path


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset PyTorch random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    

@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cleanup_gpu():
    """Clean up GPU memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()