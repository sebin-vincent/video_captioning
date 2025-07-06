"""
Attention Visualization for SwinBert Explainability.

This module provides functions to create visual representations of attention
mechanisms in the SwinBert model. It takes processed attention data (typically
from the `attention_mapper` module) and overlays it onto video frames to
highlight regions of importance.

Key functionalities include:
- Overlaying patch-level attention as heatmaps on video frames.
- Generating videos that show frame-level highlights and patch-level heatmaps
  corresponding to the model's focus for specific generated text tokens.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def highlight_frames(video_path, frame_indices, scores, output_path):
    """
    Highlights specified frames in a video.
    (This is a basic placeholder - could draw border, change color, etc.)

    Args:
        video_path: Path to the input video.
        frame_indices: List of frame indices to highlight.
        scores: Optional list of scores corresponding to frame_indices, for intensity.
        output_path: Path to save the video with highlighted frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in frame_indices:
            # Example: Draw a green border
            cv2.rectangle(frame, (0,0), (width-1, height-1), (0,255,0), 10)
            # Could use 'scores' to modulate color or thickness

        out.write(frame)
        current_frame_idx += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


def overlay_patch_attention(frame, patch_attentions, patch_size, colormap=cv2.COLORMAP_JET):
    """
    Overlays patch attention as a heatmap on a single frame.

    Args:
        frame: The video frame (numpy array).
        patch_attentions: A 2D numpy array (H_patches, W_patches) with attention scores for patches.
        patch_size: Tuple (patch_H, patch_W).
        colormap: OpenCV colormap to use for the heatmap.

    Returns:
        The frame with attention heatmap overlaid.
    """
    frame_height, frame_width, _ = frame.shape
    num_patches_h, num_patches_w = patch_attentions.shape

    # Resize attention map to frame size for overlay
    heatmap_resized = cv2.resize(patch_attentions, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255 and apply colormap
    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)

    # Blend heatmap with frame
    alpha = 0.5 # Transparency factor
    overlaid_frame = cv2.addWeighted(heatmap_colored, alpha, frame, 1 - alpha, 0)

    return overlaid_frame


def visualize_token_explanation(video_path, token_explanation_data, output_video_path, patch_size):
    """
    Creates a video visualizing the explanation for a single token.
    It will highlight key frames and overlay patch attentions on them.

    Args:
        video_path: Path to the original video.
        token_explanation_data: Data structure from attention_mapper.map_attention_to_frames_patches.
                                Example:
                                {
                                  "token": "word_x",
                                  "key_frames": [
                                      {"frame_idx": F1, "score": S1,
                                       "key_patches_map": np.array (H_patches, W_patches)}, ...
                                  ]
                                }
        output_video_path: Path to save the output video.
        patch_size: Tuple (patch_H, patch_W).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare data for easy lookup
    # token_explanation_data is a list of explanations, one for each text token.
    # This function visualizes for ONE token at a time.
    # The data structure for one token is:
    # {
    #   "token_str": "word",
    #   "top_frames": [ {"frame_idx": F_abs, "avg_attention_to_frame": A_score,
    #                    "patch_attention_map": np.array (H_patches, W_patches)}, ... ]
    # }

    key_frame_details = {
        kf_info['frame_idx']: {
            "avg_score": kf_info['avg_attention_to_frame'],
            "patch_map": kf_info['patch_attention_map']
        } for kf_info in token_explanation_data.get('top_frames', [])
    }

    current_frame_idx = 0
    max_overall_frame_score = 0
    if key_frame_details:
        max_overall_frame_score = max(d['avg_score'] for d in key_frame_details.values() if d['avg_score'] is not None)
        if max_overall_frame_score == 0: max_overall_frame_score = 1.0 # Avoid division by zero if all scores are 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = frame.copy()
        if current_frame_idx in key_frame_details:
            info = key_frame_details[current_frame_idx]
            patch_map_for_overlay = info.get('patch_map')

            if patch_map_for_overlay is not None:
                # Normalize patch map locally for this frame for better visualization
                map_min, map_max = patch_map_for_overlay.min(), patch_map_for_overlay.max()
                if map_max > map_min: # Avoid division by zero if map is flat
                    patch_map_normalized_locally = (patch_map_for_overlay - map_min) / (map_max - map_min)
                else:
                    patch_map_normalized_locally = patch_map_for_overlay - map_min # Should be all zeros

                processed_frame = overlay_patch_attention(processed_frame, patch_map_normalized_locally, patch_size)

            # Add a border, color intensity based on frame's avg_attention_to_frame score
            # relative to max score for this token.
            frame_score_norm = info['avg_score'] / max_overall_frame_score
            border_intensity = int(frame_score_norm * 255)
            border_color = (0, border_intensity, 0) # Green, intensity varies
            cv2.rectangle(processed_frame, (0,0), (width-1, height-1), border_color, 10)

            # Put text: Token being explained & current frame score
            token_text = f"Token: {token_explanation_data.get('token_str', '')}"
            frame_score_text = f"Frame {current_frame_idx} score: {info['avg_score']:.2f}"
            cv2.putText(processed_frame, token_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, frame_score_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        out.write(processed_frame)
        current_frame_idx += 1

    cap.release()
    out.release()
    print(f"Explanation video for token '{token_explanation_data.get('token_str')}' saved to {output_video_path}")


if __name__ == '__main__':
    # Example usage (will be filled out later)
    print("Visualization module created.")
    # Create a dummy video for testing
    # dummy_video_path = "dummy_video.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # temp_out = cv2.VideoWriter(dummy_video_path, fourcc, 1, (100, 100))
    # for i in range(10): # 10 frames
    #     img = np.zeros((100, 100, 3), dtype=np.uint8)
    #     cv2.putText(img, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #     temp_out.write(img)
    # temp_out.release()

    # highlight_frames(dummy_video_path, [2, 5, 8], None, "dummy_highlighted.mp4")

    # mock_frame = np.zeros((224,224,3), dtype=np.uint8)
    # mock_patch_attn = np.random.rand(14, 14) # e.g. 224/16 = 14 patches
    # overlaid = overlay_patch_attention(mock_frame, mock_patch_attn, (16,16))
    # cv2.imwrite("dummy_patch_overlay.png", overlaid)

    # mock_token_exp = {
    #     "token": "test",
    #     "key_frames": [
    #         {"frame_idx": 3, "score": 0.8, "key_patches_map": np.random.rand(7,7)}, # Assuming 224/32 patch size for Swin
    #         {"frame_idx": 7, "score": 0.9, "key_patches_map": np.random.rand(7,7)}
    #     ]
    # }
    # visualize_token_explanation(dummy_video_path, mock_token_exp, "dummy_token_explanation.mp4", (32,32))
    pass
