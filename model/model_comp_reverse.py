from ._base import *
from .model_comp import ModelComp

class ModelCompReverse(ModelComp):
    def apply_attention(self, x_A, x_B, attention_layer):
        # Reshape the inputs to be compatible with nn.MultiheadAttention
        batch_size, channels, height, width = x_A.size()
        
        # Flatten the spatial dimensions (HxW) into a sequence for MultiheadAttention
        x_A_flat = x_A.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, H*W, channels]
        x_B_flat = x_B.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, H*W, channels]

        # Apply attention
        # attn_output, _ = attention_layer(x_B_flat, x_A_flat, x_A_flat) # Q, K, V # TODO make reverse the knowledge transfer direction
        attn_output, _ = attention_layer(x_A_flat, x_B_flat, x_B_flat) # Q, K, V

        # Reshape the output back to [batch_size, channels, H, W]
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        # Add the attention output to x_B to get input for the next layer
        x_B_next = x_B + attn_output

        return x_B_next
