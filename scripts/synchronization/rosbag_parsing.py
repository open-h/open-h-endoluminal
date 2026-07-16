#!/usr/bin/env python

"""
ROS bag parser for extracting scalar calibration signals from an endoluminal
recording (endoscope camera vs. EM-tracked scope tip).

ROS versions and non-ROS rigs:
    This example targets ROS1 (rospy / rosbag). ROS 2 users can record with
    rosbag2 and adapt this parsing step; the offset-then-sync pattern is
    unchanged. Non-ROS rigs (EM tracker SDK logs, capsule telemetry) can skip
    this file and enter the pipeline at the temp_cali.py stage with any pair
    of timestamped arrays.

This script processes a ROS1 `.bag` file and extracts timestamped scalar
signals from selected topics. It prepares the inputs for temporal
calibration (temp_cali.py): wiggle the scope tip sinusoidally while
recording both the endoscope camera (/endoscope/image_raw) and the EM
tracker pose (/em_tracker/tip_pose), then reduce each stream to one scalar
per message so the two can be sine-fitted and phase-compared. The estimated
per-topic offsets (for example, frame-grabber latency on the video path)
are then applied in post_sync.py.

Core features:
--------------
- Extracts messages and timestamps from specified ROS topics.
- Organizes data in a dictionary with the format:
    - "t_<topic_name>": list of timestamps in seconds
    - "y_<topic_name>": list of associated float-encoded message data
- Saves the extracted data to disk as a Python pickle file.

Conversion functions:
---------------------
- `image_to_float`: STUB, must be implemented for your rig. Track a visual
  keypoint or marker in the endoscope view and return its pixel coordinate.
- `pose_to_float`: working example that returns the norm of a
  geometry_msgs/PoseStamped translation. Rename or reroute it if your
  tracker publishes a different message type.

Dependencies:
-------------
- ROS1 (rospy, rosbag)
- OpenCV + cv_bridge
- NumPy
- Pickle
"""


import rosbag
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


def image_to_float(bridge, msg):
    """
    Reduce an endoscope image to a single scalar for temporal calibration.

    STUB: the implementation is rig-specific, so this function raises
    NotImplementedError until you fill it in. Track a visual keypoint or
    marker in the endoscope view (for example, an ArUco marker or a
    high-contrast feature on the phantom wall, via cv2.aruco, template
    matching, or optical flow) and return one of its pixel coordinates
    (u or v) as a float. The returned scalar must oscillate with the
    scope-tip wiggle so temp_cali.py can fit a sine wave to it.
    """
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    raise NotImplementedError(
        "image_to_float is a stub: track a visual keypoint or marker in the "
        "endoscope view (cv_image above is the decoded frame) and return one "
        "of its pixel coordinates as a float."
    )


def pose_to_float(msg):
    """
    Working example: reduce a geometry_msgs/PoseStamped from the EM tracker
    to a single scalar, the Euclidean norm of the tip translation. As the
    scope tip wiggles sinusoidally, this norm oscillates at the same
    frequency, which is all temp_cali.py needs.
    """
    p = msg.pose.position
    return float(np.linalg.norm([p.x, p.y, p.z]))


def parse_bag(bag_file_path, topics_of_interest, output_pkl_path):
    '''
    Parses a ROS1 bag file and extracts data from selected topics into a
    dictionary, which is then saved as a pickle file. Each topic produces:
        - t_<topic>: list of timestamps (in seconds)
        - y_<topic>: list of message data (converted to floats where possible)
    Parameters:
    -----------
        bag_file_path (str): Path to the input .bag file.
        topics_of_interest (list of str): List of ROS topics to extract.
        output_pkl_path (str): Path to save the resulting .pkl file.
    '''

    data_dict = {}
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file_path)

    # Iterate over messages in the specified topics
    for topic, msg, t in bag.read_messages(topics=topics_of_interest):
        key_base = topic.strip("/").replace("/", "_")
        t_key = f"t_{key_base}"
        y_key = f"y_{key_base}"
        if t_key not in data_dict:
            data_dict[t_key] = []
            data_dict[y_key] = []

        # Append timestamp (converted to seconds)
        data_dict[t_key].append(t.to_sec())

        # Handle message parsing depending on type/topic
        if isinstance(msg, Image):
            data = image_to_float(bridge, msg)
        elif topic == "/em_tracker/tip_pose":
            data = pose_to_float(msg)
        else:
            # TODO: convert any other stream you want to calibrate against
            # (insertion depth, tendon displacement, ...) to a float here.
            data = msg

        # Append message data
        data_dict[y_key].append(data)

    bag.close()

    # Save to pickle
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"[INFO] Bag parsed and saved to: {output_pkl_path}")


if __name__ == "__main__":
    # Example usage. Implement image_to_float for your rig before running.
    parse_bag(
        bag_file_path="path/to/the/bagfile.bag",
        topics_of_interest=["/endoscope/image_raw", "/em_tracker/tip_pose"],
        output_pkl_path="path/to/calibration_signals.pkl",
    )
