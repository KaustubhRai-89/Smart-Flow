import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096):  # Default, override at runtime
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def forward(self, image_features):
        return self.projection(image_features)

    def save(self, path="checkpoints/projection/0.pt"):
        torch.save(self.state_dict(), path)
        print(f"Projection weights saved: {path}")

if __name__ == "__main__":
    # Dynamically fetch llm_dim via GPT-OSS config if possible
    dummy_vision = torch.randn(1, 768)
    proj = ProjectionLayer()
    output = proj(dummy_vision)
    print(f"Projection success! Output shape: {output.shape}")  # Should be [1, 4096] if changed
    proj.save()
