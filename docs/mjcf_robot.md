# MJCFRobot: MuJoCo MJCF Support

The `MJCFRobot` class provides support for loading and using robot models from MuJoCo's MJCF (MuJoCo XML Format) files, with differentiable forward kinematics powered by MuJoCo MJX.

## Installation

To use `MJCFRobot`, you need to install MuJoCo:

```bash
pip install mujoco>=3.0.0
```

## Basic Usage

```python
from pyroki import MJCFRobot
import jax.numpy as jnp

# Load robot from MJCF file
robot = MJCFRobot.from_mjcf("path/to/your/robot.xml")

# Get robot information
print(f"Joints: {robot.get_joint_names()}")
print(f"Bodies: {robot.get_body_names()}")
print(f"Joint limits: {robot.get_joint_limits()}")

# Forward kinematics
cfg = jnp.array([0.1, 0.2, 0.3])  # Joint configuration
poses = robot.forward_kinematics(cfg)  # Body poses in SE(3) format
```

## Features

### Differentiable Forward Kinematics

The `MJCFRobot` uses MuJoCo MJX for fully differentiable forward kinematics:

```python
import jax

def objective(cfg):
    poses = robot.forward_kinematics(cfg)
    # Return some function of the poses
    return jnp.sum(poses[:, 4:])  # Sum of all positions

# Compute gradients
grad_fn = jax.grad(objective)
gradient = grad_fn(cfg)
```

### Batched Computation

Forward kinematics supports batched computation for efficiency:

```python
# Batch of configurations
batch_cfg = jnp.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3],
    [0.2, 0.4, 0.6]
])

# Batched forward kinematics
batch_poses = robot.forward_kinematics(batch_cfg)
print(batch_poses.shape)  # (3, num_bodies, 7)
```

### Pose Format

The forward kinematics returns poses in SE(3) format as 7-dimensional vectors:
- `[w, x, y, z, x, y, z]` where the first 4 elements are quaternion (w, x, y, z) and the last 3 are position (x, y, z)

## API Reference

### MJCFRobot Class

#### Static Methods

- `from_mjcf(mjcf_path, default_joint_cfg=None)`: Load robot from MJCF file

#### Methods

- `forward_kinematics(cfg)`: Compute forward kinematics for given joint configuration(s)
- `get_joint_limits()`: Get joint limits as (lower_limits, upper_limits) tuple
- `get_joint_names()`: Get names of actuated joints
- `get_body_names()`: Get names of all bodies

#### Properties

- `num_actuated_joints`: Number of actuated joints
- `num_bodies`: Number of bodies in the model

## Comparison with URDF Robot

| Feature | URDF Robot | MJCF Robot |
|---------|------------|------------|
| File Format | URDF | MJCF |
| Physics Engine | Custom JAX implementation | MuJoCo MJX |
| Performance | Good for simple models | Optimized for complex models |
| Features | Basic kinematics | Full physics simulation support |
| Differentiability | ✅ | ✅ |
| Batching | ✅ | ✅ |

## Example MJCF File

Here's a simple example of an MJCF file for a 2-DOF robot:

```xml
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
```

## Limitations

- Currently supports only revolute (hinge) and prismatic (slide) joints
- Requires MuJoCo to be installed
- MJCF files must be valid MuJoCo XML format

## Performance Tips

1. Use batched computation when possible for better performance
2. JIT compile your functions that use forward kinematics
3. Consider using MuJoCo's built-in optimization features for complex models 