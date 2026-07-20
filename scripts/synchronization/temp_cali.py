#!/usr/bin/env python

"""
Estimate temporal offset between two periodic signals using sine wave fitting.

ROS versions and non-ROS rigs:
    This file is pure NumPy/SciPy and works with any pair of timestamped
    arrays. ROS1 users can produce the inputs with rosbag_parsing.py; ROS 2
    users can record with rosbag2 and adapt that parsing step (the
    offset-then-sync pattern is unchanged); non-ROS rigs (EM tracker SDK
    logs, capsule telemetry) can enter the pipeline directly here.

This script provides a function `estimate_offset` to determine the time offset between two
noisy but periodic data streams (e.g., signals from two sensors) by fitting each to a sine wave
model and comparing their phase difference. This is particularly useful when synchronizing
unsynchronized periodic signals in time-critical applications like robotics or sensor fusion.

Worked example (endoscope camera vs. EM tracker):
-------------------------------------------------
- Wiggle the scope tip sinusoidally (e.g., bend it at roughly 0.5 Hz, so
  w0 is about pi rad/s) while recording both streams.
- (t1, y1): timestamps and a tracked pixel coordinate of a keypoint or
  marker in the endoscope view (see rosbag_parsing.image_to_float).
- (t2, y2): timestamps and the norm of the EM-tracked tip translation
  (see rosbag_parsing.pose_to_float).
- estimate_offset(t1, y1, t2, y2, w0) returns the relative delay between
  the two streams (typically dominated by frame-grabber latency on the
  video path). Feed the result to post_sync.py as a per-topic time offset.

How It Works:
-------------
- Fits both y1 and y2 to sine wave models of the form A·sin(ω·t + φ) + C,
  where C is a constant offset term, so signals with a large nonzero mean
  (e.g., a pixel coordinate oscillating around the image center, or an EM
  translation norm of hundreds of millimeters with a small wiggle) do not
  need to be de-meaned before fitting.
- Estimates the phase difference between the fitted sine waves.
- Converts phase difference to time offset using: Δt = Δφ / ω.
- Returns both the principal solution and its phase-shifted alternative.

Ambiguity warning:
------------------
- Because a sine wave cannot be distinguished from its half-period-shifted
  mirror, TWO candidate offsets are returned, separated by π/ω. Disambiguate
  with a prior on the true delay (a frame grabber adds latency on the order
  of tens of milliseconds, so pick the candidate consistent with the video
  lagging the tracker), or repeat the recording at a different wiggle
  frequency: the true offset stays the same across recordings while the
  spurious candidate moves with π/ω.

Notes:
------
- Requires the signals to be reasonably well-approximated by sinusoids of the same frequency.
- Works best when noise is low and frequency ω is known and consistent.
- Useful in temporal calibration of periodic motion sensors (e.g., robot joints, images, etc.).

"""

import numpy as np
from scipy.optimize import curve_fit


def estimate_offset(t1, y1, t2, y2, w0, a1=1, a2=1):
    '''
    Estimate the temporal offset between two periodic data streams using sine fitting.
    Parameters:
    -----------
    t1 : np.ndarray of shape (N,) Time stamps of the first data stream.
    y1 : np.ndarray of shape (N,) Signal values corresponding to t1
    t2 : np.ndarray of shape (N,) Time stamps of the second data stream.
    y2 : np.ndarray of shape (N,) Signal values corresponding to t2.
    w0 : float Known angular frequency (rad/s) of the periodic motion
    a1 : float, optional (default=1) Initial amplitude guess for y1.
    a2 : float, optional (default=1) Initial amplitude guess for y2.
    Returns:
    --------
    estimated_offset_1 : float
        Primary estimated temporal offset between the two signals (in seconds).
        A positive value indicates that y2 is ahead of y1, and negative means delayed.
    estimated_offset_2 : float
        Alternate solution due to the periodic nature of sine wave fitting,
        offset by half a period (π/w) from the first solution.
    Notes:
    ------
    - The function fits both signals to sine curves with a constant offset
      term, A * sin(w * t + phi) + C, using non-linear least squares.
    - The offset term C absorbs any nonzero signal mean, so the inputs do
      not need to be de-meaned beforehand.
    - It computes the phase difference between the two fitted curves.
    - The temporal offset is inferred from the phase difference: Δt = Δφ / ω.
    - Since sine waves are periodic, two plausible offsets are returned.
    '''
    # --- Define sine model to be fitted: A * sin(w * t + φ) + C ---
    def sine_model(t, A, w, phi, C):
        return A * np.sin(w * t + phi) + C

    # --- Fit the first signal ---
    t0 = t1[0] # Shift time origin for numerical stability
    params1, _ = curve_fit(sine_model, t1-t0, y1, p0=[a1, w0, 0, np.mean(y1)])
    A1, w1, phi1, C1 = params1

    # --- Fit the second signal ---
    params2, _ = curve_fit(sine_model, t2-t0, y2, p0=[a2, w0, 0, np.mean(y2)])
    A2, w2, phi2, C2 = params2

    # --- Estimate time offset from phase difference ---
    # Δφ = w * Δt  => Δt = Δφ / w
    phase_diff = (phi2 - phi1)
    estimated_w = (w1+w2)/2

    # Normalize the phase difference to the range [-π, π] to avoid wrap-around issues
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    estimated_offset_1 = phase_diff / estimated_w

    # Due to sine periodicity, there's a second possible solution offset by ±π/w
    if estimated_offset_1 > 0:
        estimated_offset_2 = estimated_offset_1-np.pi/estimated_w
    else:
        estimated_offset_2 = estimated_offset_1+np.pi/estimated_w

    print(f"Estimated time offset:  {estimated_offset_1:.4f} or {estimated_offset_2:.4f} seconds")

    return estimated_offset_1, estimated_offset_2
