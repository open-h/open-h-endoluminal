#!/usr/bin/env python

"""
Extract and synchronize messages from a ROS bag file recorded on an
endoluminal rig (for example, an endoscope camera plus an EM-tracked
scope tip).

ROS versions and non-ROS rigs:
    This example targets ROS1 (rospy / rosbag). ROS 2 users can record with
    rosbag2 and adapt the parsing step; the offset-then-sync pattern below is
    unchanged. Non-ROS rigs (EM tracker SDK logs, capsule telemetry) can enter
    the pipeline at the temp_cali.py stage with any pair of timestamped arrays.

This tool is useful for robot learning applications where sensor streams
(endoscope video, tracked tip poses, robot state) arrive on different topics
at different rates and need to be temporally aligned before conversion to the
LeRobot format. It provides a way to:
1. Load messages from specified ROS topics in a bag file.
2. Apply optional per-topic time offsets to compensate for sensor delays
   (for example, frame-grabber latency on the endoscope video path).
3. Synchronize messages across topics based on a configurable time slop
   (tolerance).
4. Export the synchronized data for further processing or dataset generation
   (for example, in HDF5 format).

Key components:
- `SyncedRosbagExtractor`: main class that handles message loading, time
  offset correction, and greedy approximate-time synchronization.
- Uses `CvBridge` for converting ROS image messages to OpenCV format.
- The demonstration at the bottom synchronizes an endoscope image stream with
  an EM-tracked scope-tip pose stream and writes the result to HDF5.

Typical use case:
- You recorded an endoluminal navigation demonstration (phantom, ex-vivo, or
  clinical) with an endoscope camera and an EM tracker on the scope tip.
- The two sensors publish at different frequencies, and the video path adds
  latency (frame grabber, capture card), so raw timestamps are misaligned.
- You want time-synchronized tuples {image, tip pose} to build training
  episodes. Estimate the per-topic offsets first with temp_cali.py, then
  apply them here.

Usage notes:
- Requires topic types and (optionally) per-topic time offsets to be
  specified.
- Synchronization is tolerant to time differences within `slop` seconds
  (default: 0.03 s).
- TODO hooks below mark where you plug in decoding for your own topics.

Dependencies:
- `rospy`, `rosbag`, `sensor_msgs`, `geometry_msgs`, `cv_bridge`, `h5py`,
  `numpy`
"""

import h5py
import numpy as np
import rosbag
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


class SyncedRosbagExtractor:
    def __init__(self, topics=None, slop=0.03, time_offsets=None):
        """
        Initialize the rosbag extractor.
        :param topics: List of tuples (topic_name, message_type).
        :param slop: Maximum allowed time difference between messages
                     (in seconds).
        :param time_offsets: Dictionary of time offsets for each topic
                             (in seconds), added to the recorded timestamps.
        """
        topics = topics or []
        self.topics = [topic for topic, _ in topics]
        self.slop = rospy.Duration(slop)
        self.time_offsets = time_offsets if time_offsets else {}
        self.bridge = CvBridge()

    def load_messages(self, bag_path):
        """
        Load messages from the rosbag and apply time offsets.
        :param bag_path: Path to the rosbag file.
        """
        bag = rosbag.Bag(bag_path, 'r')
        self.msgs_dict = {topic: [] for topic in self.topics}
        for topic, msg, t in bag.read_messages(topics=self.topics):
            # Apply time offset (in seconds), default to 0.0
            offset = self.time_offsets.get(topic, 0.0)
            adjusted_time = t + rospy.Duration(offset)
            self.msgs_dict[topic].append((adjusted_time, msg))
        bag.close()
        # Sort messages per topic by adjusted timestamp
        for topic in self.topics:
            self.msgs_dict[topic].sort(key=lambda x: x[0])

    def synchronize_messages(self):
        """
        Greedily synchronize messages from different topics based on their
        (adjusted) timestamps.
        :return: Generator yielding tuples of synchronized messages
                 in the order of self.topics.
        """
        pointers = {topic: 0 for topic in self.topics}

        while all(pointers[topic] < len(self.msgs_dict[topic]) for topic in self.topics):
            current_msgs = {}
            current_times = {}
            for topic in self.topics:
                t, msg = self.msgs_dict[topic][pointers[topic]]
                current_msgs[topic] = msg
                current_times[topic] = t
            t_min = min(current_times.values())
            t_max = max(current_times.values())
            if (t_max - t_min) <= self.slop:
                # TODO: can be modified to also yield the timestamps, e.g.
                # yield tuple((current_times[tp], current_msgs[tp]) for tp in self.topics)
                yield tuple(current_msgs[topic] for topic in self.topics)
                for topic in self.topics:
                    pointers[topic] += 1
            else:
                # Advance the pointer of the earliest topic and try again
                for topic in self.topics:
                    if current_times[topic] == t_min:
                        pointers[topic] += 1
                        break


if __name__ == '__main__':
    '''
    An example of a rosbag info:
    types:      geometry_msgs/PoseStamped [d3812c3cbc69362b77dc0b19b345f8f5]
                sensor_msgs/Image         [060021388200f6f0f447d0fcd9c64743]
    topics:     /em_tracker/tip_pose    1858 msgs    : geometry_msgs/PoseStamped
                /endoscope/image_raw     557 msgs    : sensor_msgs/Image
    '''
    image_topic = "/endoscope/image_raw"
    pose_topic = "/em_tracker/tip_pose"

    # Per-topic time offsets in seconds, added to each recorded timestamp.
    # The endoscope video passes through a frame grabber before it reaches
    # ROS, so its stamps lag the true acquisition time. Subtracting the
    # measured latency (here 0.08 s, estimated with temp_cali.py) realigns
    # the video stream with the EM tracker.
    time_offsets = {
        image_topic: -0.08,
        pose_topic: 0.0,
    }
    topics_list = [
        (pose_topic, PoseStamped),
        (image_topic, Image),
    ]
    bag_path = 'path/to/the/bagfile.bag'
    save_path = 'path/to/the/output_episode'  # '.hdf5' is appended below

    extractor = SyncedRosbagExtractor(
        topics=topics_list, slop=0.03, time_offsets=time_offsets
    )
    extractor.load_messages(bag_path)

    # Stream the synced tuples and accumulate them into per-topic lists
    episode_dict = {topic: [] for topic, _ in topics_list}
    for synced_msgs in extractor.synchronize_messages():
        # synced_msgs is a tuple of messages in the order of topics_list
        for (topic, _), msg in zip(topics_list, synced_msgs):
            if topic == image_topic:
                img = extractor.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                episode_dict[topic].append(img)
            elif topic == pose_topic:
                p = msg.pose.position
                q = msg.pose.orientation
                episode_dict[topic].append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            else:
                # TODO: plug in decoding for your additional topics here
                # (insertion depth, tendon displacements, fluoroscopy
                # frames, teleoperation commands, ...).
                pass

    # Dump the episode into an HDF5 file
    with h5py.File(save_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as h5f:
        h5f.create_dataset(
            'observation.images.endoscope',
            data=np.asarray(episode_dict[image_topic], dtype=np.uint8),
        )
        h5f.create_dataset(
            'observation.meta.em_pose',
            data=np.asarray(episode_dict[pose_topic], dtype=np.float64),
        )
        # TODO: add your remaining streams (e.g. 'action',
        # 'observation.state') under the Open-H-Endoluminal feature names
        # before running the LeRobot conversion scripts.

    print(f"[INFO] Wrote {len(episode_dict[image_topic])} synchronized "
          f"frames to {save_path}.hdf5")
