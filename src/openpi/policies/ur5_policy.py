"""
UR5 policy input/output transforms for openpi training and inference.

This module defines how data from the UR5 robot environment maps to the model
and vice versa. These transforms are used for both training and inference.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Creates a random input example for the UR5 policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),  # 6 joints + 1 gripper
        "prompt": "prompt",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        # Convert from (C,H,W) to (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """
    Convert inputs from the UR5 environment to the model's expected format.
    
    This class is used for both training (with LeRobot dataset) and inference
    (with real robot observations).
    """

    # Determines which model will be used (PI0, PI0_FAST, or PI05)
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """
        Transform UR5 observation data to model input format.
        
        Args:
            data: Dictionary containing:
                - observation/state: (7,) array with 6 joints + 1 gripper
                - observation/image: Base/external camera image
                - observation/wrist_image: Wrist camera image
                - prompt: (optional) Language instruction
                - actions: (optional, training only) Action sequence
        
        Returns:
            Dictionary with keys: state, image, image_mask, prompt, actions
        """
        # State is already concatenated: [6 joints + 1 gripper]
        state = np.asarray(data["observation/state"])
        
        # Parse images to uint8 (H,W,C) format
        # LeRobot stores as float32 (C,H,W), but we need uint8 (H,W,C)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        
        # Create model inputs
        # PI0 and PI05 models expect: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        # PI0-FAST expects: base_0_rgb, base_1_rgb, wrist_0_rgb
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                inputs = {
                    "state": state,
                    "image": {
                        "base_0_rgb": base_image,
                        "left_wrist_0_rgb": wrist_image,
                        # Pad with zeros since we don't have a right wrist camera
                        "right_wrist_0_rgb": np.zeros_like(base_image),
                    },
                    "image_mask": {
                        "base_0_rgb": np.True_,
                        "left_wrist_0_rgb": np.True_,
                        # Mask out the padded right wrist image for PI0/PI05
                        "right_wrist_0_rgb": np.False_,
                    },
                }
            case _model.ModelType.PI0_FAST:
                inputs = {
                    "state": state,
                    "image": {
                        "base_0_rgb": base_image,
                        # Pad with zeros for second base camera
                        "base_1_rgb": np.zeros_like(base_image),
                        "wrist_0_rgb": wrist_image,
                    },
                    "image_mask": {
                        "base_0_rgb": np.True_,
                        # Don't mask padding images for PI0-FAST
                        "base_1_rgb": np.True_,
                        "wrist_0_rgb": np.True_,
                    },
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Add actions if present (training only)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        
        # Add language prompt if present
        if "prompt" in data:
            prompt = data["prompt"]
            # Handle bytes encoding from some datasets
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """
    Convert model outputs back to the UR5 environment format.
    
    This class is used during inference to extract the appropriate action
    dimensions for the UR5 robot.
    """

    def __call__(self, data: dict) -> dict:
        """
        Extract UR5 actions from model output.
        
        The model may output actions with padding. We extract only the first 7
        dimensions corresponding to the UR5's 6 joints + 1 gripper.
        
        Args:
            data: Dictionary containing model outputs with "actions" key
        
        Returns:
            Dictionary with "actions" key containing (batch, 7) array
        """
        # Only return the first 7 action dimensions (6 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}

