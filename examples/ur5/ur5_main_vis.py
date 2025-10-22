"""
UR5 Simulation Client with GUI Visualization for OpenPI Policy Server

This script runs a UR5 simulation environment with real-time GUI visualization
and connects to the OpenPI policy server for action prediction.

Usage:
    # Start policy server in Terminal 1:
    uv run scripts/serve_policy.py --env=DROID
    
    # Run this client with GUI in Terminal 2:
    python examples/ur5/ur5_main_vis.py --prompt "pick up the red cube"
"""

import collections
import dataclasses
import logging
import pathlib

import imageio
import mujoco
import mujoco.viewer
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
import time

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # UR5 environment-specific parameters
    #################################################################################################################
    scene_xml: str = "examples/ur5/ur5_scene.xml"  # Path to UR5 scene XML file
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials: int = 1  # Number of rollouts to perform
    max_steps: int = 500  # Maximum steps per episode
    
    #################################################################################################################
    # Task specification
    #################################################################################################################
    prompt: str = "pick up the blue cube"  # Language instruction for the robot

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/ur5/videos"  # Path to save videos
    seed: int = 42  # Random Seed (for reproducibility)
    camera_width: int = 224  # Camera resolution
    camera_height: int = 224


class UR5EnvWithGUI:
    """Wrapper for UR5 MuJoCo simulation environment with GUI visualization."""
    
    def __init__(self, scene_xml: str, camera_width: int = 224, camera_height: int = 224, seed: int = 42):
        """
        Initialize UR5 simulation environment with GUI.
        
        Args:
            scene_xml: Path to MuJoCo scene XML file
            camera_width: Width of camera images
            camera_height: Height of camera images
            seed: Random seed
        """
        self.scene_xml = scene_xml
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.seed = seed
        
        # Load MuJoCo model
        logging.info(f"Loading MuJoCo scene from {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        
        # Create renderer for offscreen rendering (for cameras)
        self.renderer = mujoco.Renderer(self.model, camera_height, camera_width)
        
        # Passive viewer will be created on first step
        self.viewer = None
        self._viewer_initialized = False
        
        # Set random seed
        np.random.seed(seed)
        
        logging.info(f"UR5 Environment with GUI initialized:")
        logging.info(f"  - Number of joints: {self.model.nq}")
        logging.info(f"  - Number of actuators: {self.model.nu}")
        logging.info(f"  - Number of cameras: {self.model.ncam}")
        
    def _init_viewer(self):
        """Initialize the passive viewer for GUI visualization."""
        if not self._viewer_initialized:
            try:
                # Create passive viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer_initialized = True
                logging.info("✓ GUI viewer initialized")
            except Exception as e:
                logging.warning(f"Could not initialize GUI viewer: {e}")
                logging.warning("Continuing without GUI...")
                self.viewer = None
        
    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)

        # Set to home position
        if self.model.nkey > 0:
            print(f"Resetting to home position: {self.model.key_qpos[0]}")
            self.data.qpos[:] = self.model.key_qpos[0]
            self.data.ctrl[:] = self.model.key_qpos[0][:7]
        print(f"Qpos: {self.data.qpos}")
        
        # Initialize viewer on first reset
        if not self._viewer_initialized:
            self._init_viewer()

        # Let physics settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
        return self._get_observation()
    
    def step(self, action):
        """
        Execute action in the environment and update GUI.
        
        Args:
            action: 8-dimensional action [6 joint positions + 2 gripper positions]
        
        Returns:
            observation: Dictionary containing images and robot state
            reward: Reward (always 0 for now, would need task-specific logic)
            done: Whether episode is complete (always False for now)
            info: Additional information dictionary
        """
        # Map 8D action to actuators: 6 arm joints + 2 gripper fingers
        if len(action) >= 6:
            # Set arm joint targets
            self.data.ctrl[:6] = action[:6]
            
            # Set gripper targets (if action includes gripper commands)
            if len(action) >= 8 and self.model.nu >= 8:
                # Both gripper fingers move symmetrically
                self.data.ctrl[6] = action[6]  # Left finger
                self.data.ctrl[7] = action[6]  # Right finger (mirrored)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update GUI viewer
        if self.viewer is not None:
            self.viewer.sync()
        
        # Get observation
        obs = self._get_observation()
        
        # Placeholder reward and done (would need task-specific logic)
        reward = 0.0
        done = False
        info = {}
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation from the environment."""
        # Capture external camera image
        self.renderer.update_scene(self.data, camera="external_camera_left")
        external_image = self.renderer.render()
        
        # Capture wrist camera image
        self.renderer.update_scene(self.data, camera="wrist_camera")
        wrist_image = self.renderer.render()
        
        # Get robot joint positions (first 6 are arm joints)
        joint_positions = self.data.qpos[:6].copy()
        
        # Get gripper position (7th actuator)
        gripper_position = np.array([self.data.qpos[6]]) if self.model.nq > 6 else np.array([0.0])
        
        return {
            "external_image": external_image,
            "wrist_image": wrist_image,
            "joint_positions": joint_positions,
            "gripper_position": gripper_position,
        }
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()


def run_ur5_simulation(args: Args) -> None:
    """Main function to run UR5 simulation with policy server and GUI visualization."""
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory for videos
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("UR5 Simulation with OpenPI Policy Server (GUI Enabled)")
    logging.info("=" * 60)
    logging.info(f"Task: {args.prompt}")
    logging.info(f"Policy server: {args.host}:{args.port}")
    logging.info("GUI window will open showing real-time simulation")
    
    # Initialize environment with GUI
    try:
        env = UR5EnvWithGUI(
            scene_xml=args.scene_xml,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            seed=args.seed
        )
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        logging.error("Make sure the scene XML file exists and MuJoCo is properly installed")
        return
    
    # Connect to policy server
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        logging.info("✓ Connected to policy server")
    except Exception as e:
        logging.error(f"Failed to connect to policy server: {e}")
        logging.error("Make sure the policy server is running:")
        logging.error("  uv run scripts/serve_policy.py --env=DROID")
        return
    
    # Run trials
    total_episodes = 0
    
    for trial_idx in range(args.num_trials):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Starting Trial {trial_idx + 1}/{args.num_trials}")
        logging.info(f"{'=' * 60}")
        
        # Reset environment
        obs = env.reset()
        action_plan = collections.deque()
        
        # Setup
        t = 0
        replay_images = []
        
        while t < args.max_steps + args.num_steps_wait:
            try:
                # IMPORTANT: Wait for objects to stabilize at the beginning
                if t < args.num_steps_wait:
                    # Execute zero action (maintain current position)
                    zero_action = np.zeros(8)
                    zero_action[:6] = obs["joint_positions"]  # Stay at current joint positions
                    print(f"Zero action: {zero_action}")
                    obs, reward, done, info = env.step(zero_action)
                    t += 1
                    composite_img = np.hstack([obs["external_image"], obs["wrist_image"]])
                    replay_images.append(composite_img)
                    if t % 10 == 0:
                        logging.info(f"Step {t}: Waiting for environment to stabilize...")
                    continue

                # Get images from observation
                external_img = obs["external_image"]
                wrist_img = obs["wrist_image"]
                
                # Resize images if needed
                if external_img.shape[:2] != (args.resize_size, args.resize_size):
                    external_img = image_tools.resize_with_pad(external_img, args.resize_size, args.resize_size)
                if wrist_img.shape[:2] != (args.resize_size, args.resize_size):
                    wrist_img = image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                
                # Convert to uint8
                external_img = image_tools.convert_to_uint8(external_img)
                wrist_img = image_tools.convert_to_uint8(wrist_img)
                
                # Create side-by-side composite image for video
                composite_img = np.hstack([external_img, wrist_img])
                replay_images.append(composite_img)
                
                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict in DROID format
                    element = {
                        "observation/exterior_image_1_left": external_img,
                        "observation/wrist_image_left": wrist_img,
                        "observation/joint_position": obs["joint_positions"],
                        "observation/gripper_position": obs["gripper_position"],
                        "prompt": args.prompt,
                    }
                    
                    # Query model to get action
                    result = client.infer(element)
                    action_chunk = result["actions"]
                    
                    logging.info(f"Step {t}: Received {len(action_chunk)} actions from policy")
                    logging.info(f"  First action: {action_chunk[0][:3]}... (showing first 3 dims)")
                    
                    # Add actions to plan
                    num_actions_to_use = min(len(action_chunk), args.replan_steps)
                    action_plan.extend(action_chunk[:num_actions_to_use])
                
                # Get next action from plan
                action = action_plan.popleft()
                
                # Execute action in environment (GUI updates automatically)
                obs, reward, done, info = env.step(action)
                
                if t % 20 == 0:
                    logging.info(f"Step {t}/{args.max_steps}: Action executed (watch GUI window)")
                
                t += 1
                
            except KeyboardInterrupt:
                logging.info("\nInterrupted by user")
                break
            except Exception as e:
                logging.error(f"Error at step {t}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        total_episodes += 1
        
        # Save replay video (side-by-side: external + wrist views)
        if replay_images:
            video_filename = f"ur5_vis_trial_{trial_idx}_{args.prompt.replace(' ', '_')[:30]}.mp4"
            video_path = pathlib.Path(args.video_out_path) / video_filename
            
            logging.info(f"\nSaving side-by-side video to {video_path}")
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            logging.info(f"✓ Video saved ({len(replay_images)} frames, {replay_images[0].shape[1]}x{replay_images[0].shape[0]} resolution)")
            logging.info(f"  Format: External camera (left) | Wrist camera (right)")
        
        # Log results
        logging.info(f"\nTrial {trial_idx + 1} completed:")
        logging.info(f"  - Steps executed: {t}")
        logging.info(f"  - Frames recorded: {len(replay_images)}")
    
    # Cleanup
    env.close()
    
    logging.info(f"\n{'=' * 60}")
    logging.info("Simulation Complete!")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total trials: {total_episodes}")
    logging.info(f"Videos saved to: {args.video_out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    tyro.cli(run_ur5_simulation)


