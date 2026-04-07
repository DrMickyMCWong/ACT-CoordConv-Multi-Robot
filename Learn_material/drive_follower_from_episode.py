"""Drive the follower arm through recorded actions with tolerance control.

This script loads joint targets from an ACT demonstration episode (HDF5)
and replays them on the follower arm using the Alicia SDK. Each command is
sent through `step_with_tolerance`, which mirrors the reference logic from
`record_sim_TCP_EE_depth_cam_scripted.py` to ensure the robot reaches each
pose (within tolerance) before advancing.
"""

import argparse
import os
import time
from typing import Tuple

import h5py
import numpy as np

from config.config import ROBOT_PORTS
import alicia_d_sdk

# Control loop parameters (mirrors reference implementation)
TOR_Q = 0.005          # Joint position tolerance (radians)
GRIP_TOL = 30          # Gripper tolerance in raw units (0-1000)
MAX_WAIT = 0.35        # Default maximum wait per step (seconds)
GRIP_MIN = 0           # Gripper open limit
GRIP_MAX_SAFE = 950    # Safe close limit to avoid stalls


def load_episode_actions(data_file: str, start: int = 0, end: int = None) -> np.ndarray:
    """Load action sequence from an ACT HDF5 episode."""
    with h5py.File(data_file, "r") as root:
        if "/action" not in root:
            raise KeyError("Episode file is missing '/action' dataset")
        actions = root["/action"][()]

    if end is None or end > len(actions):
        end = len(actions)
    if start < 0 or start >= end:
        raise ValueError("Invalid start/end indices for actions slice")

    return actions[start:end]


def step_with_tolerance(robot, target_pos, max_wait: float = MAX_WAIT) -> bool:
    """Send command and wait until robot is close enough or timeout expires."""
    try:
        if hasattr(target_pos, "tolist"):
            joint_targets = target_pos[:6].tolist()
        else:
            joint_targets = target_pos[:6]
        gripper_target = target_pos[6]

        gripper_target_raw = int((1 - gripper_target) * 1000)
        gripper_target_raw = int(np.clip(gripper_target_raw, GRIP_MIN, GRIP_MAX_SAFE))

        robot.set_robot_state(
            target_joints=joint_targets,
            gripper_value=gripper_target_raw,
            joint_format="rad",
            speed_deg_s=10.0,
            wait_for_completion=False,
            gripper_speed_deg_s=100,
        )

        start_time = time.time()
        while time.time() - start_time < max_wait:
            joint_gripper_state = robot.get_robot_state("joint_gripper")
            if joint_gripper_state is None:
                time.sleep(0.01)
                continue

            current_joints = np.array(joint_gripper_state.angles)
            current_gripper = joint_gripper_state.gripper

            joint_error = np.max(np.abs(current_joints - np.array(joint_targets)))
            gripper_error = abs(current_gripper - gripper_target_raw)

            if joint_error < TOR_Q and gripper_error < GRIP_TOL:
                return True

            time.sleep(0.01)

        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error in step_with_tolerance: {exc}")
        return False


def drive_episode(actions: np.ndarray, max_wait: float) -> Tuple[int, int]:
    """Connect to follower arm and replay the provided action chunk."""
    follower = alicia_d_sdk.create_robot(
        port=ROBOT_PORTS["follower"],
        gripper_type="50mm",
    )
    print("✓ Follower robot connected")

    try:
        time.sleep(2)
        successes = 0
        failures = 0
        for idx, action in enumerate(actions):
            action_list = action.astype(float).tolist()
            reached = step_with_tolerance(follower, action_list, max_wait=max_wait)
            if reached:
                successes += 1
            else:
                failures += 1
                print(f"⚠️ Step {idx} timed out; continuing to next action")
        return successes, failures
    finally:
        print("Disconnecting robot...")
        follower.disconnect()
        print("✓ Robot disconnected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay ACT episode actions on follower arm")
    parser.add_argument(
        "--data_file",
        type=str,
        default="/home/hk/Documents/ACT_Shaka/data/task1/episode_30.hdf5",
        help="Path to HDF5 episode file containing '/action' dataset",
    )
    parser.add_argument("--start", type=int, default=0, help="First action index to play")
    parser.add_argument("--end", type=int, default=None, help="One-past-last action index to play")
    parser.add_argument(
        "--max_wait",
        type=float,
        default=MAX_WAIT,
        help="Maximum seconds to wait for each action to reach tolerance",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Episode file not found: {args.data_file}")

    actions = load_episode_actions(args.data_file, start=args.start, end=args.end)
    print(f"Loaded {len(actions)} actions from {args.data_file}")

    successes, failures = drive_episode(actions, max_wait=args.max_wait)
    print(f"Replay finished. Successes: {successes}, Failures: {failures}")

if __name__ == "__main__":
    main()
