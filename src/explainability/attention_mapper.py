"""
Attention Mapper for SwinBert Explainability.

This module provides functions to process raw attention weights obtained from
the SwinTransformer3D and BERT components of a SwinBert model. It aims to
map these attentions to meaningful visual elements like video frames and image patches,
facilitating the understanding of what the model focuses on when generating captions.

Key functionalities include:
- Extracting text-to-visual attention scores from BERT.
- Processing Swin Transformer's windowed self-attention scores into patch importance metrics.
- Generating structured explanations that link text tokens to specific visual regions.

Note: Full unrolling of Swin attention to a global patch-level pairwise attention map
is complex and currently simplified to patch importance scores within windows or derived
from BERT's attention to Swin's output patches.
"""
import torch
import numpy as np

def get_bert_text_to_visual_attention(bert_attentions_all_layers, text_len, visual_len, batch_idx=0, average_heads=True):
    """
    Extracts attention from text tokens to visual tokens from the last layer of BERT.

    Args:
        bert_attentions_all_layers: List of Tensors, output from BERT encoder.
                                    Each tensor of shape (batch, num_heads, seq_len, seq_len).
                                    seq_len = text_len + visual_len.
        text_len: Length of the text sequence part.
        visual_len: Length of the visual sequence part.
        batch_idx: Index of the batch item to process.
        average_heads: If True, averages attention scores across all heads.

    Returns:
        Tensor of shape (text_len, visual_len) representing averaged attention
        from each text token to each visual token. Returns None if input is invalid.
    """
    if not bert_attentions_all_layers:
        return None

    last_layer_attentions = bert_attentions_all_layers[-1] # Shape: (B, num_heads, S, S)

    if last_layer_attentions is None:
        return None

    att_b = last_layer_attentions[batch_idx] # Shape: (num_heads, S, S)

    if average_heads:
        att_b = att_b.mean(dim=0) # Shape: (S, S)
    else:
        # If not averaging, one needs to handle multiple heads.
        # For simplicity, let's still average here or expect pre-selected head.
        # Or, the user can pass a specific head index. For now, always average.
        att_b = att_b.mean(dim=0)

    # Total sequence length S = text_len + visual_len
    # Expected full attention matrix att_b is (text_len + visual_len, text_len + visual_len)

    # Ensure the dimensions match
    if att_b.shape[0] != text_len + visual_len or att_b.shape[1] != text_len + visual_len:
        # This can happen if sequence was padded to a max length different from text_len + visual_len
        # Or if input text_len/visual_len are for the unpadded sequences.
        # Handling padding requires knowing max_seq_len and actual unpadded lengths.
        # For now, assume text_len and visual_len are the effective lengths in the attention matrix.
        # A common case: text_len is actual, visual_len is actual.
        # BERT input might be padded up to max_position_embeddings.
        # The attention matrix will be for this max_position_embeddings.
        # We need to slice based on actual text_len and visual_len.

        # This part needs to be robust. If BERT input is padded to `max_L`, then att_b is `(max_L, max_L)`.
        # We need `text_actual_len` and `visual_actual_len`.
        # Let's assume the passed text_len and visual_len are the actual, unpadded lengths.
        pass # Continue with these lengths

    text_to_visual_att = att_b[:text_len, text_len : text_len + visual_len]
    return text_to_visual_att # Shape: (text_len, visual_len)


def get_swin_patch_importance_scores(swin_attentions_all_layers_blocks, batch_idx=0, average_heads=True):
    """
    Calculates a basic importance score for each patch within its window from Swin attentions.
    This is a simplified metric and does NOT represent unrolled, global patch attention.

    Args:
        swin_attentions_all_layers_blocks: List of lists of Tensors.
            Outer list: layers. Inner list: blocks.
            Tensor shape: (B*num_windows, num_heads, N, N) where N=patches_in_window.
        batch_idx: Index of the batch item to process. (Note: Swin attentions are B*nW,
                   so this batch_idx is implicitly handled if B=1 for the input to this func).
                   This function needs to be designed carefully if B > 1 for Swin output.
                   For now, assumes effectively B=1 for the attention tensors passed.
        average_heads: If True, averages attention scores across all heads.

    Returns:
        List of lists of Tensors. Structure mirrors input.
        Each inner Tensor: (num_windows, N_patches_in_window), representing patch scores.
        Returns None if input is invalid.
    """
    if not swin_attentions_all_layers_blocks:
        return None

    processed_attentions = []
    for layer_attns in swin_attentions_all_layers_blocks:
        processed_layer_attns = []
        if not layer_attns: # Empty list for a layer
            processed_attentions.append(processed_layer_attns)
            continue

        for block_attn_tensor in layer_attns:  # (B*nW, nH, N, N)
            if block_attn_tensor is None:  # Can happen if checkpointing was used
                processed_layer_attns.append(None)
                continue

            # Assuming batch_idx is already handled by upstream slicing if B > 1 for swin_attentions,
            # or that block_attn_tensor is for a single batch item (i.e. B effectively 1 here).
            # If B > 1, one would need to select the relevant slice of B*nW.
            # E.g., if original B_swin, then block_attn_tensor[batch_idx*nW : (batch_idx+1)*nW, ...]
            # This detail is IMPORTANT and depends on how data is fed from VideoTransformer.
            # For now, let's assume block_attn_tensor is already for the correct batch_idx (e.g., B=1 overall).

            current_att = block_attn_tensor # (num_windows_for_this_item, nH, N, N)

            if average_heads:
                current_att = current_att.mean(dim=1) # (num_windows, N, N)
            else:
                # Handle multi-head case, e.g., select first head or require pre-selection
                current_att = current_att[:, 0, :, :] # Taking first head's attention

            # Importance score: sum of attention received by each patch in its window
            patch_scores = current_att.sum(dim=1)  # Sum over "attending_patch_idx" -> (num_windows, N_attended_patch)
            processed_layer_attns.append(patch_scores)
        processed_attentions.append(processed_layer_attns)

    return processed_attentions


def generate_token_visual_explanations(
    text_tokens,
    bert_text_to_visual_attention_scores, # (text_len, visual_len)
    num_frames, # Total frames in the video segment processed by Swin
    patches_per_frame_h, # Num patches vertically for one frame
    patches_per_frame_w, # Num patches horizontally for one frame
    top_k_frames_per_token=3
    ):
    """
    Generates explanations for each text token, highlighting which visual tokens (frames/patches)
    it attended to most.

    Args:
        text_tokens: List of strings (the generated caption).
        bert_text_to_visual_attention_scores: Tensor (text_len, visual_len) from get_bert_text_to_visual_attention.
                                              visual_len here is total_patches_across_frames.
        num_frames: Number of frames Swin processed.
        patches_per_frame_h: Number of patches vertically per frame.
        patches_per_frame_w: Number of patches horizontally per frame.
        top_k_frames_per_token: How many top frames to report per token.

    Returns:
        List of dicts, one for each text token:
        [
            {
                "token_str": "word",
                "top_frames": [
                    {"frame_idx": F_abs, "avg_attention_to_frame": A_score,
                     "patch_attention_map": np.array (patches_per_frame_h, patches_per_frame_w)
                    }, ...
                ]
            }, ...
        ]
    """
    if bert_text_to_visual_attention_scores is None:
        return []

    explanations = []
    text_len, total_visual_tokens = bert_text_to_visual_attention_scores.shape

    num_patches_per_frame = patches_per_frame_h * patches_per_frame_w

    # Assuming visual_tokens are flattened in order: frame0_patch0, f0_p1, ..., f1_p0, ...
    # This is a critical assumption.
    if total_visual_tokens != num_frames * num_patches_per_frame:
        # This might happen if there's an unexpected feature dimension or pooling in VideoTransformer.fc
        # Or if max_img_seq_length in VideoTransformer causes truncation/padding of vid_feats from Swin.
        print(f"Warning: Mismatch in visual token count. Expected {num_frames * num_patches_per_frame}, got {total_visual_tokens} from BERT attention.")
        # Attempt to proceed if total_visual_tokens is smaller, otherwise adjust.
        # This part needs to be very robust based on actual `vid_feats` shape fed to BERT.
        # For now, we'll cap num_frames if total_visual_tokens is smaller.
        effective_total_patches = min(total_visual_tokens, num_frames * num_patches_per_frame)
        # And we'd need to know how these visual tokens map to frames/patches.
        # The current structure of this function assumes a direct frame-by-frame, patch-by-patch flattening.
    else:
        effective_total_patches = total_visual_tokens


    for i in range(text_len):
        token_str = text_tokens[i]
        # Attention from current text token to all visual patches
        # Shape: (effective_total_patches,)
        att_to_all_patches = bert_text_to_visual_attention_scores[i, :effective_total_patches].cpu().numpy()

        frame_attentions = [] # List of (frame_idx, avg_score_to_frame, patch_map_for_frame)

        for frame_k in range(num_frames):
            start_patch_idx = frame_k * num_patches_per_frame
            end_patch_idx = (frame_k + 1) * num_patches_per_frame

            if start_patch_idx >= effective_total_patches:
                break # No more patches left to analyze for this frame

            # Attention scores for patches within this frame_k
            # Ensure we don't go out of bounds of att_to_all_patches
            patches_for_this_frame_att = att_to_all_patches[start_patch_idx : min(end_patch_idx, effective_total_patches)]

            if patches_for_this_frame_att.size == 0:
                continue

            avg_attention_to_frame = patches_for_this_frame_att.mean()

            # Create a (H_patches, W_patches) map for this frame
            patch_map = np.zeros(num_patches_per_frame)
            # Fill the available patch scores
            patch_map[:len(patches_for_this_frame_att)] = patches_for_this_frame_att
            patch_attention_map_reshaped = patch_map.reshape(patches_per_frame_h, patches_per_frame_w)

            frame_attentions.append({
                "frame_idx": frame_k, # This is relative to the segment Swin processed
                "avg_attention_to_frame": float(avg_attention_to_frame),
                "patch_attention_map": patch_attention_map_reshaped
            })

        # Sort frames by their average attention score
        frame_attentions.sort(key=lambda x: x["avg_attention_to_frame"], reverse=True)

        explanations.append({
            "token_str": token_str,
            "top_frames": frame_attentions[:top_k_frames_per_token]
        })

    return explanations


if __name__ == '__main__':
    # Example Usage (Illustrative)
    print("Attention mapper module updated.")

    # Mocking BERT attention output (last layer, averaged over heads)
    # Batch size 1, 3 text tokens, 2 frames * (2x2 patches/frame) = 8 visual tokens
    # Total sequence length for BERT = 3 (text) + 8 (visual) = 11
    mock_text_len = 3
    mock_visual_len = 8 # e.g. 2 frames, each 2x2 patches = 4 patches/frame

    # Mock attentions from text to visual part
    # (text_len, visual_len)
    mock_bert_text_to_vis_att = torch.rand(mock_text_len, mock_visual_len)

    mock_text_tokens = ["a", "dog", "runs"]

    num_f = 2 # 2 frames
    p_h, p_w = 2, 2 # Patches per frame are 2x2

    token_explanations = generate_token_visual_explanations(
        mock_text_tokens,
        mock_bert_text_to_vis_att,
        num_frames=num_f,
        patches_per_frame_h=p_h,
        patches_per_frame_w=p_w,
        top_k_frames_per_token=1
    )

    for token_exp in token_explanations:
        print(f"Token: {token_exp['token_str']}")
        for frame_info in token_exp['top_frames']:
            print(f"  Frame Index: {frame_info['frame_idx']}")
            print(f"  Avg Attention to Frame: {frame_info['avg_attention_to_frame']:.4f}")
            print(f"  Patch Attention Map (shape {frame_info['patch_attention_map'].shape}):")
            # print(frame_info['patch_attention_map']) # This would print the numpy array

    # Mocking Swin attention output
    # List[layer][block] of (num_windows, N_patches_in_window)
    # Assume 1 layer, 1 block for simplicity.
    # Assume B=1. num_windows=2, N_patches_in_window=4 (2x2 window)
    mock_swin_att_layer_block = [
        [torch.rand(2, 4)] # 2 windows, 4 patches each, score per patch
    ]
    processed_swin = get_swin_patch_importance_scores(mock_swin_att_layer_block)
    if processed_swin:
        print("\nProcessed Swin (importance per patch in window):")
        # print(processed_swin[0][0])
    pass
