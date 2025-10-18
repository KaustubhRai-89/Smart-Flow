"""
vision_language_models.py - Central Model Definitions
This file contains all custom model architectures used across the project.
"""

import torch.nn as nn

# --- Configuration ---
VISION_DIM = 768  # CLIP ViT-L/14 output dimension
LLM_DIM = 3584    # Qwen2-7B embedding dimension

# --- The "Optic Nerve": Projection Layer ---
class ProjectionLayer(nn.Module):
    """
    A robust projection layer mapping vision (CLIP) to language (Qwen2).
    This is the single source of truth for this model architecture.
    """
    def __init__(self, input_dim=VISION_DIM, output_dim=LLM_DIM, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)
