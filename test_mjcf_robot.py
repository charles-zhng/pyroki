#!/usr/bin/env python3
"""Simple test for MJCFRobot implementation."""

import tempfile
import os
import jax.numpy as jnp

def test_mjcf_robot():
    """Test the MJCFRobot implementation."""
    try:
        from pyroki import MJCFRobot
        
        # Create a simple MJCF file
        mjcf_content = """
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
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_content)
            mjcf_path = f.name
        
        try:
            # Test robot loading
            robot = MJCFRobot.from_mjcf(mjcf_path)
            print("✓ Robot loaded successfully")
            
            # Test basic properties
            assert robot.num_actuated_joints == 2
            assert robot.num_bodies == 2
            print("✓ Basic properties correct")
            
            # Test joint names
            joint_names = robot.get_joint_names()
            assert len(joint_names) == 2
            print(f"✓ Joint names: {joint_names}")
            
            # Test body names
            body_names = robot.get_body_names()
            assert len(body_names) == 2
            print(f"✓ Body names: {body_names}")
            
            # Test joint limits
            lower, upper = robot.get_joint_limits()
            assert lower.shape == (2,)
            assert upper.shape == (2,)
            print("✓ Joint limits extracted")
            
            # Test forward kinematics
            cfg = jnp.array([0.5, -0.3])
            poses = robot.forward_kinematics(cfg)
            assert poses.shape == (2, 7)  # 2 bodies, 7-dim poses
            print("✓ Forward kinematics works")
            
            # Test batched forward kinematics
            batch_cfg = jnp.array([[0.0, 0.0], [0.5, -0.3], [1.0, 0.5]])
            batch_poses = robot.forward_kinematics(batch_cfg)
            assert batch_poses.shape == (3, 2, 7)  # 3 configs, 2 bodies, 7-dim poses
            print("✓ Batched forward kinematics works")
            
            # Test differentiability
            import jax
            
            def objective(cfg):
                poses = robot.forward_kinematics(cfg)
                return jnp.sum(poses[:, 4:])  # Sum of positions
            
            grad_fn = jax.grad(objective)
            gradient = grad_fn(cfg)
            assert gradient.shape == (2,)
            print("✓ Differentiability works")
            
            print("\n🎉 All tests passed!")
            
        finally:
            # Clean up
            if os.path.exists(mjcf_path):
                os.unlink(mjcf_path)
                
    except ImportError as e:
        print(f"❌ MuJoCo not installed: {e}")
        print("Install with: pip install mujoco>=3.0.0")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_mjcf_robot() 