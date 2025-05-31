#!/usr/bin/env python3
"""Example demonstrating the MJCFRobot class with MuJoCo MJCF files."""

import jax.numpy as jnp
import numpy as np

from pyroki import MJCFRobot


def main():
    """Main example function."""
    # Example MJCF file path - you would replace this with your actual MJCF file
    # mjcf_path = "path/to/your/robot.xml"
    
    # For demonstration, let's create a simple MJCF content
    simple_mjcf_content = """
    <mujoco>
        <worldbody>
            <body name="link1" pos="0 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="box" size="0.1 0.1 0.5" rgba="1 0 0 1"/>
                <body name="link2" pos="0 0 1">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="cylinder" size="0.05 0.3" rgba="0 1 0 1"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # Save the MJCF content to a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(simple_mjcf_content)
        mjcf_path = f.name
    
    try:
        # Load the robot from MJCF
        print("Loading robot from MJCF...")
        robot = MJCFRobot.from_mjcf(mjcf_path)
        
        print(f"Robot loaded successfully!")
        print(f"Number of actuated joints: {robot.num_actuated_joints}")
        print(f"Joint names: {robot.get_joint_names()}")
        print(f"Number of bodies: {robot.num_bodies}")
        print(f"Body names: {robot.get_body_names()}")
        
        # Get joint limits
        lower_limits, upper_limits = robot.get_joint_limits()
        print(f"Joint limits:")
        for i, (name, lower, upper) in enumerate(zip(robot.get_joint_names(), lower_limits, upper_limits)):
            print(f"  {name}: [{lower:.3f}, {upper:.3f}]")
        
        # Test forward kinematics with a single configuration
        print("\nTesting forward kinematics...")
        
        # Create a test configuration
        cfg = jnp.array([0.5, -0.3])  # 2 joints
        
        # Run forward kinematics
        poses = robot.forward_kinematics(cfg)
        print(f"Single configuration shape: {poses.shape}")
        print(f"Body poses (wxyz + xyz format):")
        for i, (name, pose) in enumerate(zip(robot.get_body_names(), poses)):
            quat = pose[:4]  # w, x, y, z
            pos = pose[4:]   # x, y, z
            print(f"  {name}: quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}], "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Test batched forward kinematics
        print("\nTesting batched forward kinematics...")
        batch_size = 5
        batch_cfg = jnp.array([
            [0.0, 0.0],
            [0.5, -0.3],
            [1.0, 0.5],
            [-0.5, 0.8],
            [0.2, -0.1]
        ])
        
        batch_poses = robot.forward_kinematics(batch_cfg)
        print(f"Batch configuration shape: {batch_cfg.shape}")
        print(f"Batch poses shape: {batch_poses.shape}")
        
        # Test gradient computation (differentiability)
        print("\nTesting differentiability...")
        import jax
        
        def fk_objective(cfg):
            """Simple objective function for testing gradients."""
            poses = robot.forward_kinematics(cfg)
            # Return sum of all end-effector positions
            return jnp.sum(poses[:, 4:])  # Sum all positions
        
        # Compute gradient
        grad_fn = jax.grad(fk_objective)
        gradient = grad_fn(cfg)
        print(f"Gradient w.r.t. joint configuration: {gradient}")
        
        print("\nExample completed successfully!")
        
    except ImportError as e:
        print(f"Error: MuJoCo not installed. Please install with: pip install mujoco>=3.0.0")
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error running example: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(mjcf_path):
            os.unlink(mjcf_path)


if __name__ == "__main__":
    main() 