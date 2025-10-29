"""
Convert UR5 robot data from pickle format to LeRobot format.

This script converts raw UR5 data stored as .pkl files in episode directories
to the LeRobot dataset format used by openpi for training.

The raw data format:
  - Each episode is a subdirectory (e.g., "1028_162704")
  - Each timestep is a .pkl file with timestamped filename
  - Each pickle file contains a dict with keys:
    - image_rgb: (480, 640, 3) uint8 - base camera
    - wrist_image_rgb: (480, 640, 3) uint8 - wrist camera
    - joint_positions: (7,) float64 - joint positions
    - gripper_position: (1,) float64 - gripper position
    - control: (7,) float64 - actions (6 joints + 1 gripper)

Usage:
    uv run examples/ur5/convert_ur5_pickle_data_to_lerobot.py \
        --data_dir /scratch/wg25/datasets/real_robot_raw/pick_up_green_lego \
        --output_repo_id ur5/pick_up_green_lego \
        --output_dir /scratch/wg25/datasets/lerobot_data

Optional flags:
    --push_to_hub: Push the converted dataset to Hugging Face Hub
    --num_workers: Number of parallel workers for processing (default: 4)
"""

import os
import pickle
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def load_pickle_episode(episode_dir: Path):
    """Load all pickle files from an episode directory and return sorted data."""
    pkl_files = sorted(episode_dir.glob("*.pkl"))
    
    if not pkl_files:
        return None
    
    episode_data = {
        'images': [],
        'wrist_images': [],
        'states': [],
        'actions': [],
    }
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract RGB images
        image_rgb = data.get('image_rgb')
        wrist_image_rgb = data.get('wrist_image_rgb')
        
        # Combine joint positions and gripper into state
        # joint_positions is (7,) but might be 6 joints + 1 already combined
        # We'll use joint_positions[:-1] for joints and gripper_position separately
        joint_positions = data.get('joint_positions', np.zeros(6))
        gripper_position = data.get('gripper_position', np.array([0.0]))
        
        # State: 6 joints + 1 gripper = 7D
        if len(joint_positions) == 7:
            # Already includes everything
            state = joint_positions
        elif len(joint_positions) == 6:
            # Need to add gripper
            state = np.concatenate([joint_positions, gripper_position])
        else:
            # Fallback: pad or truncate
            state = np.concatenate([
                joint_positions[:6] if len(joint_positions) >= 6 else np.pad(joint_positions, (0, 6 - len(joint_positions))),
                gripper_position
            ])
        
        # Actions from control
        actions = data.get('control', np.zeros(7))
        
        episode_data['images'].append(image_rgb)
        episode_data['wrist_images'].append(wrist_image_rgb)
        episode_data['states'].append(state.astype(np.float32))
        
        # Ensure actions is 7D (6 joints + 1 gripper)
        if len(actions) == 7:
            episode_data['actions'].append(actions.astype(np.float32))
        elif len(actions) == 6:
            episode_data['actions'].append(
                np.concatenate([actions, gripper_position]).astype(np.float32)
            )
        else:
            # Pad or truncate
            episode_data['actions'].append(
                np.pad(actions[:7], (0, max(0, 7 - len(actions)))).astype(np.float32)
            )
    
    return episode_data


def main(
    data_dir: str,
    output_repo_id: str = "ur5/pick_up_green_lego",
    output_dir: str | None = None,
    num_workers: int = 4,
    *,
    push_to_hub: bool = False,
    fps: int = 30,
):
    """Convert UR5 pickle format episodes to LeRobot format.
    
    Args:
        data_dir: Path to directory containing episode subdirectories
        output_repo_id: Name for the output dataset
        output_dir: Custom output directory (if None, uses ~/.cache/lerobot/)
        num_workers: Number of parallel workers for episode processing
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        fps: Frames per second (default: 30)
    """
    data_path = Path(data_dir)
    
    # Find all episode directories
    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    num_episodes = len(episode_dirs)
    
    if num_episodes == 0:
        raise ValueError(f"No episode directories found in {data_dir}")
    
    print(f"Found {num_episodes} episodes")
    print(f"FPS: {fps}")
    print(f"Using {num_workers} workers for parallel processing")
    
    # Determine output path
    if output_dir is not None:
        output_path = Path(output_dir) / output_repo_id
        os.environ['LEROBOT_HOME'] = str(Path(output_dir))
        print(f"Custom output directory: {output_path}")
    else:
        output_path = HF_LEROBOT_HOME / output_repo_id
        print(f"Default output directory: {output_path}")
    
    # Clean up any existing dataset
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="ur5",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # 6 joints + 1 gripper
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 6 joints + 1 gripper
                "names": ["actions"],
            },
        },
        image_writer_threads=max(10, num_workers * 2),
        image_writer_processes=max(5, num_workers),
    )
    
    # Helper function to load episode data
    def load_episode(episode_dir):
        """Load episode data from pickle files."""
        try:
            return load_pickle_episode(episode_dir)
        except Exception as e:
            print(f"Error loading episode {episode_dir.name}: {e}")
            return None
    
    # Load episodes in parallel
    print(f"\nLoading {num_episodes} episodes in parallel...")
    episodes_data = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_dir = {
            executor.submit(load_episode, ep_dir): ep_dir 
            for ep_dir in episode_dirs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_dir):
            ep_dir = future_to_dir[future]
            result = future.result()
            if result is not None:
                episodes_data[ep_dir] = result
                print(f"  Loaded episode {ep_dir.name} ({len(result['images'])} frames)")
    
    if not episodes_data:
        raise ValueError("No episodes could be loaded successfully!")
    
    # Prepare frame dictionaries in parallel (this speeds up the sequential add_frame calls)
    print(f"\nPreparing frame data structures in parallel...")
    task_description = "pick up the green lego"
    
    def prepare_episode_frames(episode_data):
        """Prepare all frame dictionaries for an episode."""
        num_frames = len(episode_data['images'])
        frames = []
        for frame_idx in range(num_frames):
            frames.append({
                "image": episode_data['images'][frame_idx],
                "wrist_image": episode_data['wrist_images'][frame_idx],
                "state": episode_data['states'][frame_idx],
                "actions": episode_data['actions'][frame_idx],
                "task": task_description,
            })
        return frames
    
    # Prepare frames in parallel
    prepared_episodes = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_dir = {
            executor.submit(prepare_episode_frames, episode_data): ep_dir
            for ep_dir, episode_data in episodes_data.items()
        }
        
        for future in as_completed(future_to_dir):
            ep_dir = future_to_dir[future]
            frames = future.result()
            prepared_episodes[ep_dir] = frames
            print(f"  Prepared episode {ep_dir.name} ({len(frames)} frames)")
    
    # Add episodes to dataset in order (must be sequential due to LeRobot internal state)
    # But now we're just adding pre-prepared dictionaries, which is much faster
    # NOTE: LeRobot's image_writer_threads and image_writer_processes handle parallel image I/O
    #       behind the scenes, so image writing is already parallelized!
    print(f"\nAdding episodes to LeRobot dataset...")
    total_frames = 0
    for ep_idx, (ep_dir, frames) in enumerate(sorted(prepared_episodes.items())):
        num_frames = len(frames)
        
        print(f"Processing episode {ep_idx+1}/{len(prepared_episodes)} ({num_frames} frames)...")
        
        # Add pre-prepared frames (faster than building dicts on-the-fly)
        # The image_writer_threads/processes handle parallel image writing internally
        for frame in frames:
            dataset.add_frame(frame)
        
        # Save the episode
        dataset.save_episode()
        total_frames += num_frames
        print(f"  ✓ Episode {ep_idx+1} complete")
    
    print(f"\n{'='*60}")
    print(f"Dataset conversion complete!")
    print(f"{'='*60}")
    print(f"Output location: {output_path}")
    print(f"Total episodes: {len(dataset.episode_data_index)}")
    print(f"Total frames: {total_frames}")
    print(f"Average frames/episode: {total_frames / len(dataset.episode_data_index):.1f}")
    
    if output_dir is not None:
        print(f"\n⚠️  IMPORTANT: Set this environment variable before training:")
        print(f"export LEROBOT_HOME={output_dir}")
        print(f"\nOr update your config to use the full path:")
        print(f"repo_id=\"{output_dir}/{output_repo_id}\"")
    
    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["ur5", "robot", "manipulation", "pick_up_green_lego"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print("Push complete!")


if __name__ == "__main__":
    tyro.cli(main)

