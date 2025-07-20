import torch
import numpy as np

def map_attention_to_patches(attention_scores, num_frames, patch_resolution):
    """
    Maps attention scores to video frames and patches.

    Args:
        attention_scores (torch.Tensor): A tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
            containing the attention scores.
        num_frames (int): The number of frames in the video.
        patch_resolution (tuple): A tuple containing the height and width of the patch grid.

    Returns:
        A tensor of shape (batch_size, num_heads, num_frames, patch_resolution[0], patch_resolution[1])
        containing the attention scores for each patch in each frame.
    """
    # Get the attention scores for the [CLS] token
    cls_attention = attention_scores[:, :, 0, 1:]  # Exclude the [CLS] token itself

    # Reshape the attention scores to match the video frames and patches
    batch_size, num_heads, sequence_length = cls_attention.shape
    num_patches = patch_resolution[0] * patch_resolution[1]

    # Ensure that the sequence length is consistent
    expected_sequence_length = num_frames * num_patches
    if sequence_length != expected_sequence_length:
        raise ValueError(f"The sequence length of the attention scores ({sequence_length}) does not match the expected sequence length ({expected_sequence_length}).")

    attention_maps = cls_attention.view(batch_size, num_heads, num_frames, num_patches)
    attention_maps = attention_maps.view(batch_size, num_heads, num_frames, patch_resolution[0], patch_resolution[1])

    return attention_maps

def visualize_attention(attention_maps, frames, output_path):
    """
    Visualizes attention maps on video frames.

    Args:
        attention_maps (torch.Tensor): A tensor of shape (batch_size, num_heads, num_frames, patch_resolution[0], patch_resolution[1])
            containing the attention scores for each patch in each frame.
        frames (list): A list of PIL Images representing the video frames.
        output_path (str): The path to save the output video or images.
    """
    import cv2
    import matplotlib.pyplot as plt

    batch_size, num_heads, num_frames, height, width = attention_maps.shape

    # Average the attention scores across all heads
    avg_attention = torch.mean(attention_maps, dim=1)

    for i in range(batch_size):
        for j in range(num_frames):
            frame = np.array(frames[j])
            attention_map = avg_attention[i, j].cpu().numpy()

            # Resize the attention map to the frame size
            resized_attention_map = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))

            # Create a heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * resized_attention_map), cv2.COLORMAP_JET)

            # Superimpose the heatmap on the original frame
            superimposed_img = heatmap * 0.4 + frame

            # Save the frame
            output_filename = f"{output_path}_batch_{i}_frame_{j}.jpg"
            cv2.imwrite(output_filename, superimposed_img)
