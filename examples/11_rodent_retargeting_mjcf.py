#!/usr/bin/env python3
"""
Rodent Motion Retargeting using MJCFRobot

This example demonstrates how to retarget motion capture data to a rodent model
using the MJCFRobot class with MuJoCo MJX for differentiable forward kinematics.

The script loads:
- A rodent MJCF model
- Motion capture keypoints data
- Performs optimization to match robot poses to the keypoints
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from loguru import logger
from jaxopt import GradientDescent

import pyroki as pk

# Keypoint names as provided
KP_NAMES = [
    'Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase',
    'ShoulderL', 'ElbowL', 'WristL', 'HandL', 'ShoulderR', 'ElbowR', 
    'WristR', 'HandR', 'HipL', 'KneeL', 'AnkleL', 'FootL', 'HipR', 
    'KneeR', 'AnkleR', 'FootR'
]

# Mapping from keypoint names to MJCF body names
KEYPOINT_MODEL_PAIRS = {
    'AnkleL': 'lower_leg_L',
    'AnkleR': 'lower_leg_R',
    'EarL': 'skull',
    'EarR': 'skull',
    'ElbowL': 'upper_arm_L',
    'ElbowR': 'upper_arm_R',
    'FootL': 'foot_L',
    'FootR': 'foot_R',
    'HandL': 'hand_L',
    'HandR': 'hand_R',
    'HipL': 'pelvis',
    'HipR': 'pelvis',
    'KneeL': 'upper_leg_L',
    'KneeR': 'upper_leg_R',
    'ShoulderL': 'scapula_L',
    'ShoulderR': 'scapula_R',
    'Snout': 'skull',
    'SpineF': 'vertebra_cervical_5',
    'SpineL': 'pelvis',
    'SpineM': 'vertebra_1',
    'TailBase': 'pelvis',
    'WristL': 'lower_arm_L',
    'WristR': 'lower_arm_R'
}


def load_mocap_data(mocap_path: str) -> jnp.ndarray:
    """Load motion capture keypoints data.
    
    Args:
        mocap_path: Path to the .npy file containing keypoints data
        
    Returns:
        Array of shape (n_frames, n_keypoints, 3) containing 3D keypoint positions
    """
    logger.info(f"Loading mocap data from {mocap_path}")
    data = np.load(mocap_path)
    logger.info(f"Loaded mocap data shape: {data.shape}")
    
    # Ensure data is in the right format (n_frames, n_keypoints, 3)
    if len(data.shape) == 3 and data.shape[-1] == 3:
        return jnp.array(data)
    elif len(data.shape) == 2 and data.shape[1] % 3 == 0:
        n_keypoints = data.shape[1] // 3
        return jnp.array(data.reshape(-1, n_keypoints, 3))
    else:
        raise ValueError(f"Expected mocap data shape (n_frames, n_keypoints, 3) or (n_frames, n_keypoints*3), got {data.shape}")


def setup_retargeting_mapping(robot: pk.MJCFRobot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Set up mapping between keypoints and robot bodies.
    
    Args:
        robot: The MJCFRobot instance
        
    Returns:
        Tuple of (keypoint_indices, body_indices) for the mapping
    """
    body_names = robot.get_body_names()
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}
    
    keypoint_indices = []
    body_indices = []
    
    for kp_name in KP_NAMES:
        if kp_name in KEYPOINT_MODEL_PAIRS:
            body_name = KEYPOINT_MODEL_PAIRS[kp_name]
            if body_name in body_name_to_idx:
                keypoint_indices.append(KP_NAMES.index(kp_name))
                body_indices.append(body_name_to_idx[body_name])
            else:
                logger.warning(f"Body '{body_name}' not found in robot model")
        else:
            logger.warning(f"No mapping found for keypoint '{kp_name}'")
    
    logger.info(f"Set up {len(keypoint_indices)} keypoint-to-body mappings")
    return jnp.array(keypoint_indices), jnp.array(body_indices)


# Note: Using jaxopt for optimization instead of jaxls cost functions


def retarget_frame(
    robot: pk.MJCFRobot,
    target_keypoints: jnp.ndarray,
    keypoint_indices: jnp.ndarray,
    body_indices: jnp.ndarray,
    initial_cfg: jnp.ndarray,
    max_iterations: int = 100,
) -> tuple[jnp.ndarray, bool]:
    """Retarget a single frame of keypoints to robot configuration.
    
    Args:
        robot: The MJCFRobot instance
        target_keypoints: Target keypoint positions (n_keypoints, 3)
        keypoint_indices: Indices of keypoints to use
        body_indices: Corresponding robot body indices
        initial_cfg: Initial robot configuration
        max_iterations: Maximum optimization iterations
        
    Returns:
        Tuple of (optimized_config, success_flag)
    """
    # Create a simple optimization using JAX directly
    def objective(cfg):
        # Forward kinematics to get all body poses
        body_poses = robot.forward_kinematics(cfg)
        
        # Extract positions of target bodies
        target_body_poses = body_poses[body_indices]  # (n_targets, 7)
        target_positions = target_body_poses[:, 4:]   # (n_targets, 3) - xyz positions
        
        # Get corresponding target keypoints
        target_kp_positions = target_keypoints[keypoint_indices]  # (n_targets, 3)
        
        # Position error
        position_error = target_positions - target_kp_positions
        position_cost = jnp.sum(position_error ** 2)
        
        # Joint limit cost
        lower_limits, upper_limits = robot.get_joint_limits()
        limit_violations = (
            jnp.sum(jnp.maximum(0.0, cfg - upper_limits) ** 2) +
            jnp.sum(jnp.maximum(0.0, lower_limits - cfg) ** 2)
        )
        
        return position_cost * 1000.0 + limit_violations * 1e3
    
    # Use JAX optimization directly
    solver = GradientDescent(fun=objective, maxiter=max_iterations)
    result = solver.run(initial_cfg)
    
    optimized_cfg = result.params
    success = result.state.error < 1e-6  # Simple convergence check
    
    return optimized_cfg, success


def main(
    mocap_path: str = "/Users/charleszhang/GitHub/stac-mjx/coltrane_2021_07_29_1_kps.npy",
    mjcf_path: str = "examples/retarget_helpers/rodent/rodent.xml",
    start_frame: int = 0,
    num_frames: int = 10,
    max_iterations: int = 100,
    save_results: bool = True,
) -> None:
    """Main function for rodent retargeting.
    
    Args:
        mocap_path: Path to mocap keypoints data
        mjcf_path: Path to rodent MJCF file
        start_frame: Starting frame for retargeting
        num_frames: Number of frames to process
        max_iterations: Maximum optimization iterations per frame
        save_results: Whether to save results
    """
    try:
        # Load robot
        logger.info(f"Loading rodent model from {mjcf_path}")
        robot = pk.MJCFRobot.from_mjcf(mjcf_path)
        logger.info(f"Robot loaded with {robot.num_actuated_joints} joints and {robot.num_bodies} bodies")
        
        # Load mocap data
        mocap_data = load_mocap_data(mocap_path)
        n_frames, n_keypoints, _ = mocap_data.shape
        
        if n_keypoints != len(KP_NAMES):
            logger.warning(f"Expected {len(KP_NAMES)} keypoints, got {n_keypoints}")
        
        # Set up retargeting mapping
        keypoint_indices, body_indices = setup_retargeting_mapping(robot)
        
        if len(keypoint_indices) == 0:
            logger.error("No valid keypoint-to-body mappings found!")
            return
        
        # Process frames
        end_frame = min(start_frame + num_frames, n_frames)
        logger.info(f"Processing frames {start_frame} to {end_frame-1}")
        
        results = []
        success_count = 0
        
        # Get initial configuration (default)
        lower_limits, upper_limits = robot.get_joint_limits()
        initial_cfg = (lower_limits + upper_limits) / 2.0
        
        for frame_idx in range(start_frame, end_frame):
            logger.info(f"Processing frame {frame_idx}/{end_frame-1}")
            
            # Get target keypoints for this frame
            target_keypoints = mocap_data[frame_idx]
            
            # Retarget this frame
            optimized_cfg, success = retarget_frame(
                robot=robot,
                target_keypoints=target_keypoints,
                keypoint_indices=keypoint_indices,
                body_indices=body_indices,
                initial_cfg=initial_cfg,
                max_iterations=max_iterations,
            )
            
            if success:
                success_count += 1
                # Use this as initial guess for next frame
                initial_cfg = optimized_cfg
            
            results.append({
                'frame': frame_idx,
                'config': optimized_cfg,
                'success': success,
                'target_keypoints': target_keypoints,
            })
            
            logger.info(f"Frame {frame_idx}: {'✓' if success else '✗'}")
        
        logger.info(f"Successfully retargeted {success_count}/{len(results)} frames")
        
        # Save results if requested
        if save_results:
            output_path = "rodent_retargeting_results.npz"
            configs = jnp.stack([r['config'] for r in results])
            successes = jnp.array([r['success'] for r in results])
            target_kps = jnp.stack([r['target_keypoints'] for r in results])
            
            np.savez(
                output_path,
                configs=configs,
                successes=successes,
                target_keypoints=target_kps,
                keypoint_names=KP_NAMES,
                joint_names=robot.get_joint_names(),
            )
            logger.info(f"Results saved to {output_path}")
        
        # Compute and log some statistics
        if len(results) > 0:
            final_errors = []
            for result in results:
                if result['success']:
                    # Compute final error
                    body_poses = robot.forward_kinematics(result['config'])
                    target_positions = body_poses[body_indices, 4:]  # xyz positions
                    target_kp_positions = result['target_keypoints'][keypoint_indices]
                    error = jnp.linalg.norm(target_positions - target_kp_positions, axis=1)
                    final_errors.append(jnp.mean(error))
            
            if final_errors:
                mean_error = jnp.mean(jnp.array(final_errors))
                logger.info(f"Mean retargeting error: {mean_error:.4f}")
        
    except ImportError as e:
        logger.error(f"MuJoCo not installed: {e}")
        logger.error("Please install with: pip install mujoco>=3.0.0")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that the mocap data and MJCF files exist")
    except Exception as e:
        logger.error(f"Error during retargeting: {e}")
        raise


if __name__ == "__main__":
    tyro.cli(main) 