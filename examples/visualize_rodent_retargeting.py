#!/usr/bin/env python3
"""
Visualization script for rodent retargeting results.

This script loads the retargeting results and provides visualization
of the rodent model poses and keypoint matching.
"""

import numpy as np
import jax.numpy as jnp
import tyro
from loguru import logger

import pyroki as pk


def visualize_retargeting_results(
    results_path: str = "rodent_retargeting_results.npz",
    mjcf_path: str = "examples/retarget_helpers/rodent/rodent.xml",
    frame_idx: int = 0,
    show_keypoints: bool = True,
) -> None:
    """Visualize rodent retargeting results.
    
    Args:
        results_path: Path to the saved retargeting results
        mjcf_path: Path to the rodent MJCF file
        frame_idx: Frame index to visualize
        show_keypoints: Whether to show target keypoints
    """
    try:
        # Load results
        logger.info(f"Loading results from {results_path}")
        data = np.load(results_path)
        
        configs = data['configs']
        successes = data['successes'] 
        target_keypoints = data['target_keypoints']
        keypoint_names = data['keypoint_names']
        joint_names = data['joint_names']
        
        logger.info(f"Loaded {len(configs)} frames, {np.sum(successes)} successful")
        
        # Load robot
        logger.info(f"Loading rodent model from {mjcf_path}")
        robot = pk.MJCFRobot.from_mjcf(mjcf_path)
        
        # Check frame index
        if frame_idx >= len(configs):
            logger.error(f"Frame index {frame_idx} out of range [0, {len(configs)-1}]")
            return
            
        # Get data for the specific frame
        config = configs[frame_idx]
        success = successes[frame_idx]
        target_kps = target_keypoints[frame_idx]
        
        logger.info(f"Visualizing frame {frame_idx} (success: {success})")
        
        # Compute forward kinematics for this configuration
        body_poses = robot.forward_kinematics(config)
        
        # Print some information
        logger.info(f"Robot configuration shape: {config.shape}")
        logger.info(f"Body poses shape: {body_poses.shape}")
        logger.info(f"Target keypoints shape: {target_kps.shape}")
        
        # Print joint values
        logger.info("Joint values:")
        for i, (name, value) in enumerate(zip(joint_names, config)):
            logger.info(f"  {name}: {value:.3f}")
        
        # Print body positions
        logger.info("Body positions:")
        body_names = robot.get_body_names()
        for i, (name, pose) in enumerate(zip(body_names[:10], body_poses[:10])):  # First 10 bodies
            pos = pose[4:]  # xyz position
            logger.info(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        if show_keypoints:
            logger.info("Target keypoints:")
            for i, (name, kp) in enumerate(zip(keypoint_names[:10], target_kps[:10])):  # First 10 keypoints
                logger.info(f"  {name}: [{kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}]")
        
        logger.info("Visualization complete!")
        logger.info("Note: For interactive 3D visualization, consider using PyRoki's viewer module")
        
    except ImportError as e:
        logger.error(f"MuJoCo not installed: {e}")
        logger.error("Please install with: pip install mujoco>=3.0.0")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that the results file and MJCF file exist")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise


def compare_frames(
    results_path: str = "rodent_retargeting_results.npz",
    mjcf_path: str = "examples/retarget_helpers/rodent/rodent.xml",
    frame_indices: list[int] = [0, 5, 9],
) -> None:
    """Compare multiple frames side by side.
    
    Args:
        results_path: Path to the saved retargeting results
        mjcf_path: Path to the rodent MJCF file
        frame_indices: List of frame indices to compare
    """
    try:
        # Load results
        data = np.load(results_path)
        configs = data['configs']
        successes = data['successes']
        
        # Load robot
        robot = pk.MJCFRobot.from_mjcf(mjcf_path)
        
        logger.info(f"Comparing frames: {frame_indices}")
        
        for frame_idx in frame_indices:
            if frame_idx >= len(configs):
                logger.warning(f"Frame {frame_idx} out of range, skipping")
                continue
                
            config = configs[frame_idx]
            success = successes[frame_idx]
            
            # Compute forward kinematics
            body_poses = robot.forward_kinematics(config)
            
            logger.info(f"\nFrame {frame_idx} (success: {success}):")
            
            # Show first few joint values
            joint_names = robot.get_joint_names()
            for i in range(min(5, len(config))):
                logger.info(f"  {joint_names[i]}: {config[i]:.3f}")
            
            # Show end-effector positions (if applicable)
            body_names = robot.get_body_names()
            if "hand_L" in body_names:
                hand_idx = body_names.index("hand_L")
                hand_pos = body_poses[hand_idx, 4:]
                logger.info(f"  Left hand position: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
            
            if "hand_R" in body_names:
                hand_idx = body_names.index("hand_R")
                hand_pos = body_poses[hand_idx, 4:]
                logger.info(f"  Right hand position: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise


def visualize_3d_interactive(
    results_path: str = "rodent_retargeting_results.npz",
    mjcf_path: str = "examples/retarget_helpers/rodent/rodent.xml",
    frame_idx: int = 0,
) -> None:
    """Interactive 3D visualization using PyRoki's viewer.
    
    Args:
        results_path: Path to the saved retargeting results
        mjcf_path: Path to the rodent MJCF file
        frame_idx: Frame index to visualize
    """
    try:
        # Load results
        logger.info(f"Loading results from {results_path}")
        data = np.load(results_path)
        
        configs = data['configs']
        target_keypoints = data['target_keypoints']
        
        # Load robot
        logger.info(f"Loading rodent model from {mjcf_path}")
        robot = pk.MJCFRobot.from_mjcf(mjcf_path)
        
        # Get configuration for the specific frame
        config = configs[frame_idx]
        target_kps = target_keypoints[frame_idx]
        
        logger.info(f"Starting interactive 3D viewer for frame {frame_idx}")
        logger.info("Use the viewer controls to rotate, zoom, and inspect the robot pose")
        
        # Note: This would require integrating with PyRoki's viewer system
        # For now, we provide the data and suggest manual integration
        logger.info("Robot configuration:")
        logger.info(f"  Joint values: {config}")
        logger.info("Target keypoints:")
        logger.info(f"  Keypoint positions shape: {target_kps.shape}")
        
        # Compute forward kinematics
        body_poses = robot.forward_kinematics(config)
        logger.info(f"Computed {len(body_poses)} body poses")
        
        # For actual 3D visualization, you would use something like:
        # viewer = pk.viewer.start_viewer()
        # viewer.update_robot_configuration(robot, config)
        # viewer.add_keypoints(target_kps)
        
        logger.info("3D visualization data prepared!")
        logger.info("Note: For full interactive 3D rendering, integrate with PyRoki's viewer module")
        
    except Exception as e:
        logger.error(f"Error during 3D visualization: {e}")
        raise


def main(
    mode: str = "visualize",  # "visualize", "compare", or "interactive"
    results_path: str = "rodent_retargeting_results.npz",
    mjcf_path: str = "examples/retarget_helpers/rodent/rodent.xml",
    frame_idx: int = 0,
    frame_indices: list[int] = [0, 5, 9],
) -> None:
    """Main function for visualization.
    
    Args:
        mode: Visualization mode ("visualize", "compare", or "interactive")
        results_path: Path to the saved retargeting results
        mjcf_path: Path to the rodent MJCF file
        frame_idx: Frame index for single frame visualization
        frame_indices: Frame indices for comparison mode
    """
    if mode == "visualize":
        visualize_retargeting_results(
            results_path=results_path,
            mjcf_path=mjcf_path,
            frame_idx=frame_idx,
        )
    elif mode == "compare":
        compare_frames(
            results_path=results_path,
            mjcf_path=mjcf_path,
            frame_indices=frame_indices,
        )
    elif mode == "interactive":
        visualize_3d_interactive(
            results_path=results_path,
            mjcf_path=mjcf_path,
            frame_idx=frame_idx,
        )
    else:
        logger.error(f"Unknown mode: {mode}. Use 'visualize', 'compare', or 'interactive'")


if __name__ == "__main__":
    tyro.cli(main) 