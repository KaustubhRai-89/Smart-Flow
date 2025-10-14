import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=5120):  # CLIP dim → GPT-OSS dim (adjust if needed)
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),  # Main transformation
            nn.GELU(),                       # Activation for non-linearity
            nn.Linear(llm_dim, llm_dim),     # Extra layer for better alignment
            nn.LayerNorm(llm_dim)            # Normalize for stability
        )

    def forward(self, image_features):
        return self.projection(image_features)

# Test it quickly
if __name__ == "__main__":
    proj = ProjectionLayer()
    dummy_vision = torch.randn(1, 768)  # Fake embedding like from CLIP
    output = proj(dummy_vision)
    print(f"Projection success! Output shape: {output.shape}")  # Should be [1, 5120]
