# Rodent Motion Retargeting with MJCFRobot

This example demonstrates how to retarget motion capture data to a rodent model using PyRoki's `MJCFRobot` class with MuJoCo MJX for differentiable forward kinematics.

## Files

- `11_rodent_retargeting_mjcf.py`: Main retargeting script
- `visualize_rodent_retargeting.py`: Visualization script for results
- `retarget_helpers/rodent/rodent.xml`: Rodent MJCF model

## Prerequisites

1. **Install MuJoCo**: The example requires MuJoCo for the MJCFRobot class:
   ```bash
   pip install mujoco>=3.0.0
   ```

2. **Motion Capture Data**: The example expects keypoint data at:
   ```
   /Users/charleszhang/GitHub/stac-mjx/coltrane_2021_07_29_1_kps.npy
   ```
   
   The data should be in format `(n_frames, n_keypoints, 3)` where `n_keypoints = 23`.

## Keypoint Mapping

The script uses the following mapping between motion capture keypoints and rodent body parts:

| Keypoint | Body Part | MJCF Body |
|----------|-----------|-----------|
| Snout | Head | skull |
| EarL/EarR | Ears | skull |
| SpineF/M/L | Spine | vertebra_cervical_5, vertebra_1, pelvis |
| ShoulderL/R | Shoulders | scapula_L/R |
| ElbowL/R | Elbows | upper_arm_L/R |
| WristL/R | Wrists | lower_arm_L/R |
| HandL/R | Hands | hand_L/R |
| HipL/R | Hips | pelvis |
| KneeL/R | Knees | upper_leg_L/R |
| AnkleL/R | Ankles | lower_leg_L/R |
| FootL/R | Feet | foot_L/R |
| TailBase | Tail | pelvis |

## Usage

### Basic Retargeting

Run the retargeting on the first 10 frames:

```bash
python examples/11_rodent_retargeting_mjcf.py
```

### Custom Parameters

Specify custom parameters:

```bash
python examples/11_rodent_retargeting_mjcf.py \
    --mocap-path /path/to/your/keypoints.npy \
    --mjcf-path examples/retarget_helpers/rodent/rodent.xml \
    --start-frame 0 \
    --num-frames 50 \
    --max-iterations 200
```

### Parameters

- `--mocap-path`: Path to motion capture keypoints (.npy file)
- `--mjcf-path`: Path to rodent MJCF model file
- `--start-frame`: Starting frame index (default: 0)
- `--num-frames`: Number of frames to process (default: 10)
- `--max-iterations`: Maximum optimization iterations per frame (default: 100)
- `--save-results`: Whether to save results (default: True)

## Visualization

After running the retargeting, visualize the results:

```bash
# Visualize a single frame
python examples/visualize_rodent_retargeting.py --mode visualize --frame-idx 5

# Compare multiple frames
python examples/visualize_rodent_retargeting.py --mode compare --frame-indices [0,5,9]
```

## Output

The script saves results to `rodent_retargeting_results.npz` containing:

- `configs`: Optimized joint configurations (n_frames, n_joints)
- `successes`: Success flags for each frame (n_frames,)
- `target_keypoints`: Original keypoint data (n_frames, n_keypoints, 3)
- `keypoint_names`: Names of keypoints
- `joint_names`: Names of robot joints

## Technical Details

### Optimization Process

For each frame, the script:

1. **Forward Kinematics**: Computes robot body poses using MJCFRobot
2. **Cost Function**: Minimizes position error between robot bodies and keypoints
3. **Constraints**: Enforces joint limits
4. **Solver**: Uses Gauss-Newton optimization via jaxls

### Cost Function

The retargeting cost function minimizes:

```python
position_error = robot_body_positions - target_keypoints
cost = ||position_error||²
```

With additional joint limit constraints to keep solutions physically valid.

### Differentiability

The entire pipeline is differentiable thanks to:
- MuJoCo MJX for differentiable physics
- JAX for automatic differentiation
- jaxls for differentiable optimization

## Performance Tips

1. **Batch Processing**: Process multiple frames for better temporal consistency
2. **Warm Starting**: Use previous frame's solution as initial guess
3. **Iteration Limits**: Adjust `max_iterations` based on accuracy vs. speed needs
4. **Frame Selection**: Start with a subset of frames to test parameters

## Troubleshooting

### Common Issues

1. **MuJoCo Import Error**: Install MuJoCo with `pip install mujoco>=3.0.0`
2. **File Not Found**: Check that mocap data and MJCF files exist
3. **Shape Mismatch**: Verify mocap data has 23 keypoints in expected format
4. **Convergence Issues**: Try increasing `max_iterations` or adjusting weights

### Debugging

- Enable verbose logging to see detailed progress
- Check that body names in MJCF match the expected mapping
- Visualize intermediate results to identify issues

## Extending the Example

### Custom Keypoint Mappings

Modify `KEYPOINT_MODEL_PAIRS` to use different body mappings:

```python
KEYPOINT_MODEL_PAIRS = {
    'custom_keypoint': 'custom_body_name',
    # ... add your mappings
}
```

### Additional Costs

Add regularization or smoothness costs:

```python
# Smoothness cost between frames
@jaxls.Cost.create_factory
def smoothness_cost(
    vals: jaxls.VarValues,
    joint_var: jaxls.Var[jax.Array],
    prev_joint_var: jaxls.Var[jax.Array],
    weight: float = 1.0,
) -> jax.Array:
    """Smoothness cost between consecutive frames."""
    return (vals[joint_var] - vals[prev_joint_var]) * weight
```

### Different Optimization

Use alternative solvers or add custom constraints as needed.

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PyRoki Documentation](https://chungmin99.github.io/pyroki/)
- [JAX Documentation](https://jax.readthedocs.io/) 