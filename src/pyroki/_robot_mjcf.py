"""MuJoCo MJCF Robot implementation using MJX for differentiable forward kinematics."""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import jaxls
import mujoco
import mujoco.mjx as mjx
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float


@jdc.pytree_dataclass
class MJCFRobot:
    """A differentiable robot kinematics tree using MuJoCo MJX."""

    mjx_model: jdc.Static[mjx.Model]
    """MJX model for the robot."""

    mj_model: jdc.Static[mujoco.MjModel]
    """Original MuJoCo model for accessing metadata."""

    joint_names: jdc.Static[tuple[str, ...]]
    """Names of actuated joints."""

    body_names: jdc.Static[tuple[str, ...]]
    """Names of all bodies in the model."""

    joint_indices: jdc.Static[tuple[int, ...]]
    """Indices of actuated joints in the MuJoCo model."""

    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]
    """Variable class for the robot configuration."""

    @staticmethod
    def from_mjcf(
        mjcf_path: str,
        default_joint_cfg: Float[ArrayLike, "*batch actuated_count"] | None = None,
    ) -> MJCFRobot:
        """
        Loads a robot kinematic tree from an MJCF file.

        Args:
            mjcf_path: Path to the MJCF file.
            default_joint_cfg: The default joint configuration to use for optimization.
        """
        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        mjx_model = mjx.put_model(mj_model)

        # Get actuated joint information
        joint_names = []
        joint_indices = []
        
        for i in range(mj_model.njnt):
            joint_type = mj_model.jnt_type[i]
            if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                joint_name = mj_model.joint(i).name
                if not joint_name:
                    joint_name = f"joint_{i}"
                joint_names.append(joint_name)
                joint_indices.append(i)

        joint_names = tuple(joint_names)
        joint_indices = tuple(joint_indices)

        # Get body names (skip world body)
        body_names = []
        for i in range(1, mj_model.nbody):  # Skip world body (index 0)
            body_name = mj_model.body(i).name
            if not body_name:
                body_name = f"body_{i}"
            body_names.append(body_name)
        body_names = tuple(body_names)

        # Compute default joint configuration
        num_actuated = len(joint_names)
        if default_joint_cfg is None:
            # Use joint range midpoints as default
            joint_ranges = []
            for joint_idx in joint_indices:
                jnt_range = mj_model.jnt_range[joint_idx]
                if jnt_range[0] == jnt_range[1]:  # No limits
                    joint_ranges.append(0.0)
                else:
                    joint_ranges.append((jnt_range[0] + jnt_range[1]) / 2.0)
            default_joint_cfg = jnp.array(joint_ranges)
        else:
            default_joint_cfg = jnp.array(default_joint_cfg)

        assert default_joint_cfg.shape == (num_actuated,)

        # Variable class for the robot configuration.
        class JointVar(
            jaxls.Var[Array],
            default_factory=lambda: default_joint_cfg,
        ): ...

        robot = MJCFRobot(
            mjx_model=mjx_model,
            mj_model=mj_model,
            joint_names=joint_names,
            body_names=body_names,
            joint_indices=joint_indices,
            joint_var_cls=JointVar,
        )

        return robot

    @jdc.jit
    def forward_kinematics(
        self,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch body_count 7"]:
        """Run forward kinematics on the robot's bodies, in the provided configuration.

        Computes the world pose of each body frame. The result is ordered
        corresponding to `self.body_names`.

        Args:
            cfg: The configuration of the actuated joints, in the format `(*batch actuated_count)`.

        Returns:
            The SE(3) transforms of the bodies, ordered by `self.body_names`,
            in the format `(*batch, body_count, wxyz_xyz)`.
        """
        batch_axes = cfg.shape[:-1]
        num_actuated = len(self.joint_names)
        assert cfg.shape == (*batch_axes, num_actuated)

        # Handle batched computation
        if len(batch_axes) == 0:
            # Single configuration
            return self._forward_kinematics_single(cfg)
        else:
            # Batched configurations
            flat_cfg = cfg.reshape(-1, num_actuated)
            flat_results = jax.vmap(self._forward_kinematics_single)(flat_cfg)
            return flat_results.reshape(*batch_axes, len(self.body_names), 7)

    def _forward_kinematics_single(
        self, cfg: Float[Array, "actuated_count"]
    ) -> Float[Array, "body_count 7"]:
        """Forward kinematics for a single configuration."""
        # Create MJX data with the given configuration
        mjx_data = mjx.make_data(self.mjx_model)
        
        # Create full qpos array (MuJoCo may have more DOFs than just our actuated joints)
        qpos = jnp.zeros(self.mjx_model.nq)
        
        # Map our joint configuration to the appropriate positions in qpos
        for i, joint_idx in enumerate(self.joint_indices):
            qpos_addr = self.mjx_model.jnt_qposadr[joint_idx]
            qpos = qpos.at[qpos_addr].set(cfg[i])
        
        # Set joint positions
        mjx_data = mjx_data.replace(qpos=qpos)
        
        # Run forward kinematics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        
        # Extract body poses (position + quaternion) - skip world body
        body_positions = mjx_data.xpos[1:]  # Skip world body
        body_quaternions = mjx_data.xquat[1:]  # Skip world body, format: [w, x, y, z]
        
        # Convert to SE(3) format [w, x, y, z, x, y, z]
        poses = jnp.concatenate([
            body_quaternions,  # [w, x, y, z]
            body_positions     # [x, y, z]
        ], axis=-1)
        
        return poses

    def get_joint_limits(self) -> tuple[Float[Array, "actuated_count"], Float[Array, "actuated_count"]]:
        """Get joint limits for actuated joints.
        
        Returns:
            Tuple of (lower_limits, upper_limits) arrays.
        """
        lower_limits = []
        upper_limits = []
        
        for joint_idx in self.joint_indices:
            jnt_range = self.mj_model.jnt_range[joint_idx]
            lower_limits.append(jnt_range[0])
            upper_limits.append(jnt_range[1])
        
        return jnp.array(lower_limits), jnp.array(upper_limits)

    def get_body_names(self) -> tuple[str, ...]:
        """Get the names of all bodies in the model."""
        return self.body_names

    def get_joint_names(self) -> tuple[str, ...]:
        """Get the names of all actuated joints in the model."""
        return self.joint_names

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the model."""
        return len(self.body_names)

    @property
    def num_actuated_joints(self) -> int:
        """Number of actuated joints in the model."""
        return len(self.joint_names) 