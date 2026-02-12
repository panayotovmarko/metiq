#!/usr/bin/env python

"""media_analyze.py module description."""


import numpy as np
import pandas as pd
import os
import json
import subprocess

import audio_parse
import media_parse
import vft
import video_parse
import time
import sys


# Global thresholds for audio and A/V sync quality analysis
THRESHOLDS = {
    "audio": {
        "acceleration_threshold_ms_per_sec": 1,  # 1ms/sec clock error (300ms in 5 min)
        "total_wrong_speed_threshold_ms": 100,  # 100ms total drift (ITU-R detectable range)
        "jitter_threshold_ms": 30,  # 30ms jitter (stddev)
        "sudden_jump_threshold_ms": 100,  # 100ms jump (>3% of 3sec period)
        "missing_beeps_percentage_threshold": 5.0,  # >5% missing is a warning
    },
    "video": {
        "frame_drops_percentage_threshold": 1.0,  # >1% dropped frames is a warning
        "frame_duplicates_percentage_threshold": 0.5,  # >0.5% duplicates is an error
        "timing_jitter_threshold_multiplier": 0.15,  # Jitter stddev > 15% of mean spacing
        "vft_read_errors_percentage_threshold": 5.0,  # >5% read errors is a warning
        "frame_rate_variation_threshold_fps": 2.0,  # >2 fps variation indicates inconsistency
        "frame_rate_window_sec": 1.0,  # Window for frame rate calculation
    },
    "avsync": {
        # ITU-R BT.1359-1 standard thresholds (independent of frame rate)
        "itu_r_detectable_min_ms": -125,
        "itu_r_detectable_max_ms": 45,
        "itu_r_acceptable_min_ms": -185,
        "itu_r_acceptable_max_ms": 90,
        # Drift thresholds
        "drift_rate_threshold_ms_per_sec": 10,  # 10ms/sec is noticeable
        "drift_total_threshold_ms": 50,  # 50ms total drift is significant
        # Jitter/variance threshold (frame-rate aware)
        "jitter_base_threshold_ms": 30,  # Base: at least 30ms
        # Sudden jump threshold (frame-rate aware): Must be > 1 frame time
        "sudden_jump_frame_multiplier": 2.0,  # Multiplier: 2 frame times
        # Periodic oscillation: Autocorrelation strength threshold
        "oscillation_correlation_threshold": 0.5,  # Require strong correlation
        "oscillation_min_period": 5,  # Ignore very short periods (likely noise)
        # Non-monotonic drift: Require significant slope changes
        "drift_direction_min_slope": 0.001,  # Minimum slope (1ms per sample)
        # Outlier detection: IQR multiplier
        "outlier_iqr_multiplier": 1.5,  # Standard IQR outlier detection
        "outlier_percentage_threshold": 5.0,  # Flag if >5% are outliers
        # Temporal segmentation (frame-rate aware)
        "temporal_avg_change_base_ms": 50,  # Base: 50ms
        "temporal_avg_change_frame_multiplier": 1.5,  # Multiplier: 1.5 frames
        "temporal_std_change_base_ms": 20,  # Base: 20ms
        "temporal_std_change_frame_multiplier": 1.0,  # Multiplier: 1 frame
    },
}


def calculate_value_read_smoothed(video_results, ref_fps=30):
    """
    Calculate smoothed value_read, only correcting frames with reading errors.

    This function detects reading errors by checking local consistency:
    - If a frame's value_read doesn't match the expected value based on timestamp
    - If a frame's value_read creates a gap (prev+2 == next but curr != prev+1)

    Only frames with detected errors are corrected using timestamp-based calculation.
    Frames with consistent value_read sequences are left unchanged.

    The algorithm runs multiple passes because fixing one error may reveal another
    (e.g., a duplicate followed by a shifted value).
    """
    # Work with a copy to avoid modifying during iteration
    video_results = video_results.copy()

    # Start with original values
    video_results["value_read_smoothed"] = video_results["value_read"].copy()

    # Get valid (non-null) value_read entries for fitting
    valid_mask = video_results["value_read"].notna()
    valid_data = video_results[valid_mask]

    if len(valid_data) < 3:
        # Not enough data to detect errors
        return video_results

    # Calculate the expected value based on timestamp
    # value_read = ref_fps * timestamp + intercept
    timestamps = valid_data["timestamp"].values
    values = valid_data["value_read"].values
    residuals = values - ref_fps * timestamps
    # Use median instead of mean to be robust against outliers from
    # frozen frames, black frames, or other distortions
    intercept = np.median(residuals)

    # Calculate expected values for all frames
    expected_values = np.round(ref_fps * video_results["timestamp"] + intercept)

    # Run multiple passes to catch cascading errors (e.g., duplicate followed by shift)
    n = len(video_results)
    max_passes = 5

    for pass_num in range(max_passes):
        corrections_made = 0
        # Use smoothed values for neighbor checks (important for cascading errors)
        smoothed = video_results["value_read_smoothed"].values.copy()

        for i in range(1, n - 1):
            if pd.isna(smoothed[i]):
                continue

            prev_val = smoothed[i - 1] if not pd.isna(smoothed[i - 1]) else None
            next_val = smoothed[i + 1] if not pd.isna(smoothed[i + 1]) else None
            curr_val = smoothed[i]

            if prev_val is None or next_val is None:
                continue

            expected_val = int(expected_values[i])

            # Check for reading errors by comparing against expected value.
            # A frame is considered an error if it doesn't match the
            # timestamp-based expected value AND the neighbors suggest
            # a local inconsistency.

            # 1. Value doesn't match expected and neighbors are consistent
            is_error = (curr_val != expected_val) and (
                prev_val == int(expected_values[i - 1])
                or next_val == int(expected_values[i + 1])
            )

            # 2. Gap: prev and next differ by 2 but current doesn't fit
            is_gap = (next_val - prev_val == 2) and (curr_val != prev_val + 1)

            if is_error or is_gap:
                corrected_val = expected_val

                if corrected_val != curr_val:
                    video_results.at[video_results.index[i], "value_read_smoothed"] = (
                        corrected_val
                    )
                    corrections_made += 1

        if corrections_made == 0:
            break  # No more corrections needed

    # Reorder columns to place value_read_smoothed right after value_read
    cols = list(video_results.columns)
    value_read_idx = cols.index("value_read")
    cols.remove("value_read_smoothed")
    cols.insert(value_read_idx + 1, "value_read_smoothed")
    video_results = video_results[cols]

    return video_results


def calculate_frames_moving_average(video_results, windowed_stats_sec):
    # frame, ts, video_result_frame_num_read_int
    video_results = pd.DataFrame(video_results.dropna(subset=["value_read"]))

    if len(video_results) == 0:
        return pd.DataFrame()
    # only one testcase and one situation so no filter is needed.
    startframe = video_results.iloc[0]["value_read"]
    endframe = video_results.iloc[-1]["value_read"]

    frame = int(startframe)
    window_sum = 0
    tmp = 0
    average = []
    while frame < endframe:
        current = video_results.loc[video_results["value_read"].astype(int) == frame]

        if len(current) == 0:
            frame += 1
            continue
        nextframe = video_results.loc[
            video_results["timestamp"]
            >= (current.iloc[0]["timestamp"] + windowed_stats_sec)
        ]
        if len(nextframe) == 0:
            break

        nextframe_num = nextframe.iloc[0]["value_read"]

        windowed_data = video_results.loc[
            (video_results["value_read"] >= frame)
            & (video_results["value_read"] < nextframe_num)
        ]
        window_sum = len(np.unique(windowed_data["value_read"]))
        distance = nextframe_num - frame
        drops = distance - window_sum
        average.append(
            {
                "frame": frame,
                "frames": distance,
                "shown": window_sum,
                "drops": drops,
                "window": (
                    nextframe.iloc[0]["timestamp"] - current.iloc[0]["timestamp"]
                ),
            }
        )
        frame += 1

    return pd.DataFrame(average)


def calculate_frame_durations(video_results):
    # Calculate how many times a source frame is shown in capture frames/time
    video_results = pd.DataFrame(video_results.replace([np.inf, -np.inf], np.nan))
    video_results = video_results.dropna(subset=["value_read"])

    ref_fps, capture_fps = video_parse.estimate_fps(video_results)
    video_results["value_read_int"] = video_results["value_read"].astype(int)
    capt_group = video_results.groupby("value_read_int")
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg["frame_count"] = np.arange(1, len(cg) + 1)
    cg["time"] = cg["frame_count"] / capture_fps
    cg["capture_fps"] = capture_fps
    cg["ref_fps"] = ref_fps
    return cg


def calculate_measurement_quality_stats(audio_results, video_results):
    stats = {}

    # Count only unrecoverable errors
    readable = [
        (val in (vft.VFTReading.single_graycode.value, vft.VFTReading.ok.value))
        for val in video_results["status"]
    ].count(True)
    video_frames_capture_total = len(video_results)
    frame_errors = video_frames_capture_total - readable

    stats["video_frames_metiq_errors_percentage"] = round(
        100 * frame_errors / video_frames_capture_total, 2
    )

    # video metiq errors
    for vft_error in vft.VFTReading:
        stats["video_frames_metiq_error." + vft_error.name] = len(
            video_results.loc[video_results["status"] == vft_error.value]
        )

    # Audio signal
    audio_duration = audio_results["timestamp"].max() - audio_results["timestamp"].min()
    audio_sig_detected = len(audio_results)
    if audio_sig_detected == 0:
        audio_sig_detected = -1  # avoid division by zero
    stats["signal_distance_sec"] = audio_duration / audio_sig_detected
    stats["max_correlation"] = audio_results["correlation"].max()
    stats["min_correlation"] = audio_results["correlation"].min()
    stats["mean_correlation"] = audio_results["correlation"].mean()
    stats["index"] = 0

    return pd.DataFrame(stats, index=[0])


def calculate_stats(
    audio_latency_results,
    video_latency_results,
    avsync_list,
    video_results,
    audio_duration_samples,
    audio_duration_seconds,
    inputfile,
    debug=False,
):
    stats = {}
    ignore_latency = False
    if len(avsync_list) == 0 or len(video_results) == 0:
        print(f"Failure - no data")
        return None, None

    # 1. basic file statistics
    stats["file"] = inputfile
    video_frames_capture_duration = (
        video_results["timestamp"].max() - video_results["timestamp"].min()
    )
    stats["video_frames_capture_duration_sec"] = video_frames_capture_duration
    video_frames_capture_total = (
        video_results["frame_num"].max() - video_results["frame_num"].min()
    )
    stats["video_frames_capture_total"] = video_frames_capture_total
    stats["audio_frames_capture_duration_frames"] = audio_duration_seconds
    stats["audio_frames_capture_duration_samples"] = audio_duration_samples

    # 2. video latency statistics
    stats["video_latency_sec.num_samples"] = len(video_latency_results)
    stats["video_latency_sec.mean"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.mean(video_latency_results["video_latency_sec"])
    )
    stats["video_latency_sec.std_dev"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.std(video_latency_results["video_latency_sec"].values)
    )

    # 3. video latency statistics
    stats["audio_latency_sec.num_samples"] = len(audio_latency_results)
    stats["audio_latency_sec.mean"] = (
        np.nan
        if len(video_latency_results) == 0
        else np.mean(audio_latency_results["audio_latency_sec"])
    )
    stats["audio_latency_sec.std_dev"] = (
        np.nan
        if len(audio_latency_results) == 0
        else np.std(audio_latency_results["audio_latency_sec"].values)
    )

    # 4. avsync statistics
    stats["avsync_sec.num_samples"] = len(avsync_list)
    stats["avsync_sec.mean"] = np.mean(avsync_list["avsync_sec"])
    stats["avsync_sec.std_dev"] = np.std(avsync_list["avsync_sec"].values)

    # 5. video source (metiq) stats
    video_results["value_read_int"] = video_results["value_read"].dropna().astype(int)
    dump_frame_drops(video_results, inputfile)
    # 5.1. which source (metiq) frames have been show
    video_frames_sources_min = int(video_results["value_read_int"].min())
    video_frames_sources_max = int(video_results["value_read_int"].max())
    stats["video_frames_source_min"] = video_frames_sources_min
    stats["video_frames_source_max"] = video_frames_sources_max
    (
        video_frames_source_count,
        video_frames_source_unseen,
    ) = calculate_dropped_frames_stats(video_results)
    stats["video_frames_source_total"] = video_frames_source_count
    stats["video_frames_source_seen"] = (
        video_frames_source_count - video_frames_source_unseen
    )
    stats["video_frames_source_unseen"] = video_frames_source_unseen
    stats["video_frames_source_unseen_percentage"] = round(
        100 * video_frames_source_unseen / video_frames_source_count, 2
    )
    # 6. metiq processing statistics
    # TODO(chema): use video.csv information to calculate errors
    # stats["video_frames_metiq_errors"] = video_frames_metiq_errors
    # stats["video_frames_metiq_errors_percentage"] = round(
    #    100 * video_frames_metiq_errors / video_frames_capture_total, 2
    # )
    # video metiq errors
    # for vft_error in vft.VFTReading:
    #     stats["video_frames_metiq_error." + vft_error.name] = len(
    #         video_results.loc[video_results["status"] == vft_error.value]
    #     )
    # 7. calculate consecutive frame distribution
    capt_group = video_results.groupby("value_read_int")  # .count()
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg = cg.reset_index()
    # 7.2. times each source (metiq) frame been show
    stats["video_frames_source_appearances.mean"] = capt_group.size().mean()
    stats["video_frames_source_appearances.std_dev"] = capt_group.size().std()

    # TODO match gaps with source frame numbers?
    return pd.DataFrame(stats, columns=stats.keys(), index=[0]), cg


# Searches for the video_results row whose timestamp is closer to
# `audio_timestamp`.
# Returns a tuple containing:
# (a) the frame_num of the selected video row,
# (b) the searched (input, video) timestamp,
# (c) the matched (audio) timestamp,
# (d) the value read in the the selected video frame,
# (e) the frame_num of the next frame where a beep is expected,
# (f) the latency (assuming the initial frame_time).
def match_video_to_audio_timestamp(
    audio_timestamp,
    video_results,
    beep_period_frames,
    frame_time,
    previous_matches,
    used_video_timestamps,
    closest=False,
    match_distance_frames=-1,
    debug=1,
):
    video_results = video_results.copy()

    # Filter out already-used video timestamps to prevent reuse
    video_results = video_results[
        ~video_results["timestamp"].isin(used_video_timestamps)
    ]

    if len(video_results) == 0:
        print(
            f"Warning. No unused video frames available for audio timestamp {audio_timestamp}"
        )
        return None

    # The algorithm works as follows:
    # 1) find the frame in video that match the closest to audio_timestamp
    # 2) check the value parsed and compare to the expected beep frame time
    #    given the value just read.
    # 3) find the frame matching the beep number
    # 4) if the match in (1) was not exact adjust for it.

    # Determine which value_read column to use for matching
    # Prefer value_read_smoothed if available, as it corrects for occasional
    # VFT code reading errors
    value_read_col = "value_read"
    if "value_read_smoothed" in video_results.columns:
        value_read_col = "value_read_smoothed"

    # Limit errors (audio offset and errors)
    if match_distance_frames < 0:
        # somewhat arbitrary +/- 1 frame i.e. 33ms at 30fps
        match_distance_frames = 4

    closematch = None
    # Just find the closes match to the timestamp
    video_results["distance"] = np.abs(video_results["timestamp"] - audio_timestamp)
    # The purpose of this is just to find the beep source
    closematch = video_results.loc[video_results["distance"] < beep_period_frames]

    # remove non valid values
    closematch = closematch.loc[closematch[value_read_col].notna()]
    if len(closematch) == 0:
        print(f"Warning. No match for {audio_timestamp} within a beep period is found")
        return None

    # sort by time difference
    closematch = closematch.sort_values("distance")
    closematch.bfill(inplace=True)
    best_match = closematch.iloc[0]

    # 2) Check the value parsed and compare to the expected beep frame
    matched_value_read = best_match[value_read_col]
    # estimate the frame for the next beep based on the frequency
    next_beep_frame = (
        int(matched_value_read / beep_period_frames) + 1
    ) * beep_period_frames
    if (
        next_beep_frame - matched_value_read > beep_period_frames / 2
        and (next_beep_frame - beep_period_frames) not in previous_matches
    ):
        next_beep_frame -= beep_period_frames

    if next_beep_frame in previous_matches:
        # This one has already been seen, this is latency beyond a beep
        if debug > 0:
            print("Warning. latency beyond beep period.")
        next_beep_frame += beep_period_frames

    # Find the beep
    if closest:
        video_results["distance_frames"] = np.abs(
            video_results[value_read_col] - next_beep_frame
        )
        closematch = video_results.loc[
            video_results["distance_frames"] < match_distance_frames
        ]
    else:
        video_results["distance_frames"] = (
            video_results[value_read_col] - next_beep_frame
        )
        video_results.sort_values("distance_frames", inplace=True)
        closematch = video_results.loc[
            (video_results["distance_frames"] >= 0)
            & (video_results["distance_frames"] < match_distance_frames)
        ]

    # remove non valid values
    closematch = closematch.loc[closematch[value_read_col].notna()]
    if len(closematch) == 0:
        print(f"Warning. No match for {audio_timestamp} is found")
        return None

    # sort by frame distance, then by timestamp for ties (pick earliest frame)
    closematch = closematch.sort_values(["distance_frames", "timestamp"])
    closematch.bfill(inplace=True)
    best_match = closematch.iloc[0]

    # When using smoothed values, calculate simple time difference
    # avsync_sec = video_timestamp - audio_timestamp
    # Positive values mean audio is earlier than video
    video_timestamp = best_match["timestamp"]
    latency = video_timestamp - audio_timestamp

    # Get the smoothed value if available, otherwise use original
    value_read_smoothed = (
        best_match[value_read_col]
        if value_read_col == "value_read_smoothed"
        else best_match["value_read"]
    )

    # Find the closest frame to the expected beep

    if not closest and latency < 0:
        if debug > 0:
            print("ERROR: negative latency")
    else:
        vlat = [
            best_match["frame_num"],
            video_timestamp,
            audio_timestamp,
            best_match[
                "value_read"
            ],  # Keep original value_read in output for reference
            value_read_smoothed,  # Smoothed value used for matching
            latency,
        ]
        return vlat
    return None


def calculate_audio_latency(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # audio is {sample, ts, cor}
    # video is (frame, ts, expected, status, read, delta)
    # audio latency is the time between two correlated values where one should be higher

    prev = None
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps
    # run audio_results looking for audio latency matches,
    # defined as 2x audio correlations that are close and
    # where the correlation value goes down
    audio_latency_results = pd.DataFrame(
        columns=[
            "audio_sample1",
            "timestamp1",
            "audio_sample2",
            "timestamp2",
            "audio_latency_sec",
            "cor1",
            "cor2",
        ],
    )

    for index in range(len(audio_results)):
        if prev is not None:
            match = audio_results.iloc[index]
            ts_diff = match["timestamp"] - prev["timestamp"]
            # correlation indicates that match is an echo (if ts_diff < period)
            if not ignore_match_order and prev["correlation"] < match["correlation"]:
                # This skip does not move previoua but the next iteration will
                # test agains same prev match
                continue
            # ensure the 2x correlations are close enough
            if ts_diff >= beep_period_sec * 0.5:
                # default 3 sec -> 1.5 sec, max detected audio delay
                prev = match
                continue
            audio_latency_results.loc[len(audio_latency_results.index)] = [
                prev["audio_sample"],
                prev["timestamp"],
                match["audio_sample"],
                match["timestamp"],
                ts_diff,
                prev["correlation"],
                match["correlation"],
            ]
        prev = audio_results.iloc[index]
    # Remove echoes.
    audio_latency_results["diff"] = audio_latency_results["timestamp1"].diff()
    too_close = len(
        audio_latency_results.loc[audio_latency_results["diff"] < beep_period_sec * 0.5]
    )
    if too_close > 0:
        print(f"WARNING. Potential echoes detected - {too_close} counts")
    audio_latency_results.fillna(beep_period_sec, inplace=True)
    audio_latency_results = audio_latency_results.loc[
        audio_latency_results["diff"] > beep_period_sec * 0.5
    ]
    audio_latency_results = audio_latency_results.drop(columns=["diff"])
    return audio_latency_results


def filter_echoes(audiodata, beep_period_sec, margin):
    """
    The DataFrame audiodata have a timestamp in seconds, margin is 0 to 1.

    Filter everything that is closer than margin * beep_period_sec
    This puts the limit on the combined length of echoes in order not
    to prevent identifying the first signal too.
    """

    audiodata["timestamp_diff"] = audiodata["timestamp"].diff()
    # keep first signal even if it could be an echo - we cannot tell.
    audiodata.fillna(beep_period_sec, inplace=True)
    return audiodata.loc[audiodata["timestamp_diff"] > beep_period_sec * margin]


def calculate_video_relation(
    audio_results,
    video_results,
    audio_anchor,
    closest_reference,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video is (frame, ts, expected, status, read, delta)
    # video latency is the time between the frame shown when a signal is played
    # and the time when it should be played out
    prev = None
    video_latency_results = []
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps

    previous_matches = []
    used_video_timestamps = set()  # Track used video timestamps to prevent reuse
    video_latency_results = pd.DataFrame(
        columns=[
            "frame_num",
            "video_timestamp",
            "audio_timestamp",
            "video_value_read",
            "video_value_read_smoothed",
            "video_latency_sec",
        ],
    )

    for index in range(len(audio_results)):
        match = audio_results.iloc[index]
        # calculate video latency based on the
        # timestamp of the first (prev) audio match
        # vs. the timestamp of the video frame.
        audio_timestamp = match[audio_anchor]
        vmatch = match_video_to_audio_timestamp(
            audio_timestamp,
            video_results,
            beep_period_frames,
            frame_time,
            previous_matches,
            used_video_timestamps,
            closest=closest_reference,
        )

        if vmatch is not None:
            video_latency_results.loc[len(video_latency_results.index)] = vmatch
            previous_matches.append(vmatch[4])  # Use smoothed value for tracking
            used_video_timestamps.add(vmatch[1])  # Track video timestamp
        else:
            print(f"ERROR: no match found for video latency calculation")

    return video_latency_results


def calculate_video_latency(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video latency is the time between the frame shown when a signal is played
    # In the case of a transmission we look at the time from the first played out source
    # and when it is shown on the screen on the rx side.
    return calculate_video_relation(
        audio_results,
        video_results,
        "timestamp",
        False,
        beep_period_sec=beep_period_sec,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )


def calculate_avsync(
    audio_results,
    video_results,
    beep_period_sec,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # av sync is the difference between when a signal is heard and when the frame is shown
    # If there is a second signal, use that one.
    timefield = "timestamp2"
    if timefield not in audio_results.columns:
        timefield = "timestamp"
    avsync_results = calculate_video_relation(
        audio_results,
        video_results,
        timefield,
        True,
        beep_period_sec=beep_period_sec,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )
    avsync_results = avsync_results.rename(columns={"video_latency_sec": "avsync_sec"})
    return avsync_results


def z_filter_function(data, field, z_val):
    mean = data[field].mean()
    std = data[field].std()
    return data.drop(data[data[field] > mean + z_val * std].index)


def create_output_filename(input_filename, analysis_name):
    # We either have a XX.mov/mp4 or a XX.mov.video.csv
    name = input_filename
    if name[-10:].lower() == ".video.csv":
        name = name[:-10]
    name = f"{name}{MEDIA_ANALYSIS[analysis_name][2]}"
    return name


def all_analysis_function(**kwargs):
    outfile = kwargs.get("outfile", None)
    if not outfile:
        outfile = kwargs.get("input_video", None)

    for analysis_name in MEDIA_ANALYSIS:
        if analysis_name == "all":
            # prevent a loop :)
            continue
        kwargs["outfile"] = create_output_filename(outfile, analysis_name)
        results = MEDIA_ANALYSIS[analysis_name][0](**kwargs)


def audio_latency_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "audio_latency")

    audio_latency_results = calculate_audio_latency(
        audio_results,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    if len(audio_latency_results) > 0:
        audio_latency_results.to_csv(outfile, index=False)
    else:
        if debug > 0:
            print("Warning. No audio latency results")


def remove_non_doubles(audio_results, clean_audio):
    # Find all echoes and match with the original signal
    # If any clean signal has a more than one match, remove the furthest
    # match from the audio_results.

    residue = pd.concat([audio_results, clean_audio]).drop_duplicates(keep=False)
    closest = []
    for index, match in clean_audio.iterrows():
        closest_match = -1
        try:
            closest_match = (
                residue.loc[residue.index > index]["timestamp"] - match["timestamp"]
            ).idxmin()
        except:
            # could be that there are no signals > index for the actual ts
            pass
        closest.append(closest_match)

    # Find matches with multiple references
    multis = {}
    drop = []
    for source_index, matching_index in enumerate(closest):
        if closest.count(matching_index) > 1:
            first = clean_audio.iloc[source_index]["timestamp"]
            second = None
            try:
                second = residue.loc[residue.index == matching_index][
                    "timestamp"
                ].values[0]
                diff = abs(first - second)
                if matching_index in multis:
                    match = multis[matching_index]
                    if diff < match[1]:
                        # remove the previous match
                        drop.append(clean_audio.index[match[0]])
                        multis[matching_index] = (source_index, diff)
                    else:
                        # remove this match
                        clean_audio.drop(
                            clean_audio.index[[source_index]], inplace=True
                        )
                        drop.append(clean_audio.index[source_index])
                else:
                    # First match for this row
                    multis[matching_index] = (source_index, diff, first)

            except Exception as ex:
                print(f"ERROR: not match for residue (remove non doubles)")
    return clean_audio.drop(drop)


def video_latency_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    z_filter = kwargs.get("z_filter")
    outfile = kwargs.get("outfile")

    if len(audio_results) == 0:
        print("Warning. No audio signals present")
        return

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "video_latency")

    # Assuming that the source frame is played out when the audio signal
    # is first heard, video latency is the difference between the video frame
    # of the soruce and video frame shown on rx

    # First filter all echoes and keep only source signal
    clean_audio = filter_echoes(audio_results, beep_period_sec, 0.7)

    signal_ratio = len(clean_audio) / len(audio_results)
    if len(clean_audio) == 0:
        print("Warning. No source signals present")
        return
    elif signal_ratio < 1:
        print("Warning: unmatched echo/source signal. Removing unmatched.")
        clean_audio = remove_non_doubles(audio_results, clean_audio)

    # calculate the video latencies
    video_latency_results = calculate_video_latency(
        clean_audio,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    # filter the video latencies
    if z_filter > 0:
        video_latency_results = z_filter_function(
            video_latency_results, "video_latency_sec", z_filter
        )

    if len(video_latency_results) > 0:
        video_latency_results.to_csv(outfile, index=False)
    else:
        if debug > 0:
            print("Warning. No video latency results")


def audio_stats_function(
    audio_results,
    output_stats,
    source_file=None,
    beep_period_sec=3.0,
    total_duration_sec=None,
):
    """
    Calculate and write audio quality statistics to JSON file.

    Args:
        audio_results: DataFrame containing audio beep detection results with columns:
                       'audio_sample', 'timestamp', 'correlation'
        output_stats: Path to output JSON file for statistics
        source_file: Optional source filename for reference in JSON output
        beep_period_sec: Expected period between beeps in seconds (default 3.0)
        total_duration_sec: Total audio duration in seconds (for missing beeps detection)
    """
    if audio_results is None or len(audio_results) == 0:
        print("No audio results to write statistics")
        return None

    # Calculate inter-beep spacing
    timestamps = audio_results["timestamp"].values
    inter_beep_spacing = np.diff(timestamps)

    # Basic statistics
    spacing_average = np.mean(inter_beep_spacing)
    spacing_stddev = np.std(inter_beep_spacing)
    spacing_min = np.min(inter_beep_spacing)
    spacing_max = np.max(inter_beep_spacing)

    # Correlation statistics
    correlation_average = np.mean(audio_results["correlation"])
    correlation_min = np.min(audio_results["correlation"])

    issues = []
    issue_details = {}

    # 1. Wrong speed detection - clock rate error
    # Calculate deviation from expected spacing
    spacing_deviation_sec = beep_period_sec - spacing_average

    # Calculate acceleration rate as fractional clock error
    acceleration_rate = spacing_deviation_sec / beep_period_sec

    # Convert to acceleration in ms/sec (how much audio clock drifts per second of playback)
    acceleration_ms_per_sec = acceleration_rate * 1000

    # Calculate total duration and total clock error
    total_duration_sec = len(inter_beep_spacing) * beep_period_sec
    total_wrong_speed_ms = acceleration_ms_per_sec * total_duration_sec

    if (
        abs(acceleration_ms_per_sec)
        > THRESHOLDS["audio"]["acceleration_threshold_ms_per_sec"]
        or abs(total_wrong_speed_ms)
        > THRESHOLDS["audio"]["total_wrong_speed_threshold_ms"]
    ):
        issues.append("wrong_speed")
        issue_details["wrong_speed"] = {
            "total_duration_sec": float(total_duration_sec),
            "num_beeps": len(audio_results),
            "expected_spacing_sec": beep_period_sec,
            "actual_avg_spacing_sec": float(spacing_average),
            "acceleration_rate": float(acceleration_rate),
            "acceleration_ms_per_sec": float(acceleration_ms_per_sec),
            "threshold_acceleration_ms_per_sec": THRESHOLDS["audio"][
                "acceleration_threshold_ms_per_sec"
            ],
            "total_wrong_speed_ms": float(total_wrong_speed_ms),
            "threshold_total_wrong_speed_ms": THRESHOLDS["audio"][
                "total_wrong_speed_threshold_ms"
            ],
            "description": "Audio clock running at wrong speed",
        }

    # 2. High jitter
    jitter_ms = spacing_stddev * 1000
    if jitter_ms > THRESHOLDS["audio"]["jitter_threshold_ms"]:
        issues.append("high_jitter")
        issue_details["high_jitter"] = {
            "stddev_ms": float(jitter_ms),
            "threshold_ms": THRESHOLDS["audio"]["jitter_threshold_ms"],
            "description": "High variance in inter-beep timing indicating unstable audio",
        }

    # 3. Sudden jumps in inter-beep spacing
    if len(inter_beep_spacing) > 1:
        spacing_diff = np.diff(inter_beep_spacing)
        jump_threshold_sec = THRESHOLDS["audio"]["sudden_jump_threshold_ms"] / 1000.0

        large_jumps = np.abs(spacing_diff) > jump_threshold_sec
        num_jumps = large_jumps.sum()

        if num_jumps > 0:
            issues.append("sudden_jumps")
            jump_indices = np.where(large_jumps)[0]

            # Report jump locations
            jump_details = []
            for idx in jump_indices[:10]:  # First 10 only
                jump_info = {
                    "beep_index": int(idx + 1),  # Index of beep after jump
                    "timestamp_sec": float(timestamps[idx + 1]),
                    "jump_magnitude_ms": float(abs(spacing_diff[idx]) * 1000),
                }
                jump_details.append(jump_info)

            issue_details["sudden_jumps"] = {
                "count": int(num_jumps),
                "max_jump_ms": float(np.max(np.abs(spacing_diff)) * 1000),
                "jump_locations": jump_details,
                "threshold_ms": THRESHOLDS["audio"]["sudden_jump_threshold_ms"],
                "description": "Discontinuous jumps in inter-beep timing",
            }

    # 4. Missing beeps (warning)
    if total_duration_sec:
        expected_beeps = int(total_duration_sec / beep_period_sec)
        actual_beeps = len(audio_results)
        missing_beeps = expected_beeps - actual_beeps
        missing_percentage = (
            (missing_beeps / expected_beeps * 100) if expected_beeps > 0 else 0
        )

        if (
            missing_percentage
            > THRESHOLDS["audio"]["missing_beeps_percentage_threshold"]
        ):
            issues.append("missing_beeps")
            issue_details["missing_beeps"] = {
                "expected_beeps": expected_beeps,
                "actual_beeps": actual_beeps,
                "missing_count": missing_beeps,
                "missing_percentage": float(missing_percentage),
                "threshold_percentage": THRESHOLDS["audio"][
                    "missing_beeps_percentage_threshold"
                ],
                "description": "Beeps not detected, possibly due to dropouts or detection failures",
            }

    # Separate into errors and warnings
    warning_types = {"missing_beeps"}
    errors = [issue for issue in issues if issue not in warning_types]
    warnings = [issue for issue in issues if issue in warning_types]

    error_details = {k: v for k, v in issue_details.items() if k not in warning_types}
    warning_details = {k: v for k, v in issue_details.items() if k in warning_types}

    # Generate summary
    if len(errors) == 0 and len(warnings) == 0:
        summary = "ok"
    else:
        summary = errors + warnings

    # Return statistics dictionary (will be added to main stats)
    return {
        "summary": summary,
        "inter_beep_spacing_sec": {
            "average": float(spacing_average),
            "stddev": float(spacing_stddev),
            "min": float(spacing_min),
            "max": float(spacing_max),
            "expected": beep_period_sec,
            "size": len(inter_beep_spacing),
        },
        "correlation": {
            "average": float(correlation_average),
            "min": float(correlation_min),
        },
        "errors": error_details,
        "warnings": warning_details,
    }


def video_stats_function(
    avsync_results,
    output_stats,
    source_file=None,
    ref_fps=30.0,
):
    """
    Calculate and write video quality statistics to JSON file.

    Args:
        avsync_results: DataFrame containing avsync analysis results with columns:
                        'frame_num', 'video_timestamp', 'audio_timestamp',
                        'video_value_read', 'video_value_read_smoothed', 'avsync_sec'
        output_stats: Path to output JSON file for statistics
        source_file: Optional source filename for reference in JSON output
        ref_fps: Reference frame rate (default 30.0)
    """
    if avsync_results is None or len(avsync_results) == 0:
        print("No avsync results to write video statistics")
        return None

    # Calculate frame time for reference
    frame_time_ms = (1000.0 / ref_fps) if ref_fps > 0 else 33.33

    issues = []
    issue_details = {}

    # Clean data - only work with successfully read frames (where video_value_read_smoothed is not NaN)
    valid_frames = avsync_results.dropna(subset=["video_value_read_smoothed"])
    total_frames = len(avsync_results)
    valid_count = len(valid_frames)

    if valid_count == 0:
        return {
            "summary": ["no_valid_frames"],
            "errors": {
                "no_valid_frames": {
                    "description": "No valid VFT frames could be read from the video"
                }
            },
            "warnings": {},
        }

    # 1. VFT Read Errors (WARNING) - frames where video_value_read is NaN
    read_errors = avsync_results["video_value_read"].isna().sum()
    read_error_percentage = (read_errors / total_frames) * 100

    if (
        read_error_percentage
        > THRESHOLDS["video"]["vft_read_errors_percentage_threshold"]
    ):
        issues.append("vft_read_errors")
        issue_details["vft_read_errors"] = {
            "error_count": int(read_errors),
            "total_frames": int(total_frames),
            "error_percentage": float(read_error_percentage),
            "threshold_percentage": THRESHOLDS["video"][
                "vft_read_errors_percentage_threshold"
            ],
            "description": "High percentage of VFT barcode read failures indicating capture quality issues",
        }

    # 2. Frame Duplicates (ERROR) - consecutive identical value_read values
    value_read_values = valid_frames["video_value_read_smoothed"].values
    frame_differences = np.diff(value_read_values)

    duplicates = (frame_differences == 0).sum()
    duplicate_percentage = (
        (duplicates / (valid_count - 1)) * 100 if valid_count > 1 else 0
    )

    if (
        duplicate_percentage
        > THRESHOLDS["video"]["frame_duplicates_percentage_threshold"]
    ):
        # Find duplicate runs
        duplicate_runs = []
        current_run = 0
        for i, diff in enumerate(frame_differences):
            if diff == 0:
                current_run += 1
            else:
                if current_run > 0:
                    duplicate_runs.append(
                        {
                            "frame_num": int(valid_frames.iloc[i]["frame_num"]),
                            "value_read": int(value_read_values[i]),
                            "run_length": current_run + 1,  # +1 for the original frame
                        }
                    )
                current_run = 0
        if current_run > 0:  # Handle last run
            duplicate_runs.append(
                {
                    "frame_num": int(valid_frames.iloc[-1]["frame_num"]),
                    "value_read": int(value_read_values[-1]),
                    "run_length": current_run + 1,
                }
            )

        issues.append("frame_duplicates")
        issue_details["frame_duplicates"] = {
            "duplicate_count": int(duplicates),
            "duplicate_percentage": float(duplicate_percentage),
            "max_run_length": (
                int(max([r["run_length"] for r in duplicate_runs]))
                if duplicate_runs
                else 0
            ),
            "duplicate_runs": duplicate_runs[:10],  # First 10 runs
            "threshold_percentage": THRESHOLDS["video"][
                "frame_duplicates_percentage_threshold"
            ],
            "description": "Same VFT frame shown multiple times consecutively",
        }

    # 3. Frame Timing Jitter (ERROR)
    if "video_timestamp" in valid_frames.columns and len(valid_frames) > 1:
        timestamps = valid_frames["video_timestamp"].values
        inter_frame_times = np.diff(timestamps)

        # Filter out large gaps caused by missing VFT reads (measurement errors)
        # Use median as robust estimate of expected beep period
        median_spacing = np.median(inter_frame_times)
        # Keep only spacings within 1.5x the median (filter out gaps from missing reads)
        valid_spacings = inter_frame_times[inter_frame_times < median_spacing * 1.5]

        if len(valid_spacings) < 2:
            # Not enough valid spacings to calculate jitter
            pass
        else:
            # Calculate jitter as stddev of valid inter-frame times
            mean_inter_frame_sec = np.mean(valid_spacings)
            jitter_sec = np.std(valid_spacings)

            mean_inter_frame_ms = mean_inter_frame_sec * 1000
            jitter_ms = jitter_sec * 1000

            # For avsync data, frames are sampled at beep intervals (~3000ms for 3sec spacing)
            # not at frame rate intervals (~33ms). Use percentage of mean spacing as threshold.
            jitter_threshold_percentage = THRESHOLDS["video"][
                "timing_jitter_threshold_multiplier"
            ]
            jitter_threshold_ms = mean_inter_frame_ms * jitter_threshold_percentage

            if jitter_ms > jitter_threshold_ms:
                issues.append("timing_jitter")
                issue_details["timing_jitter"] = {
                    "jitter_stddev_ms": float(jitter_ms),
                    "mean_inter_frame_ms": float(mean_inter_frame_ms),
                    "threshold_ms": float(jitter_threshold_ms),
                    "threshold_percentage": float(jitter_threshold_percentage * 100),
                    "description": "High variance in frame presentation timing indicating poor vsync or timing stability",
                }

    # 4. Frame Rate Consistency (ERROR)
    if "video_timestamp" in valid_frames.columns and len(valid_frames) > ref_fps:
        window_sec = THRESHOLDS["video"]["frame_rate_window_sec"]
        frame_rates = []

        # Calculate frame rate in sliding windows
        for i in range(len(valid_frames)):
            start_time = valid_frames.iloc[i]["video_timestamp"]
            end_time = start_time + window_sec

            # Count frames in this window
            frames_in_window = valid_frames[
                (valid_frames["video_timestamp"] >= start_time)
                & (valid_frames["video_timestamp"] < end_time)
            ]

            if len(frames_in_window) > 1:
                actual_duration = (
                    frames_in_window.iloc[-1]["video_timestamp"]
                    - frames_in_window.iloc[0]["video_timestamp"]
                )
                if actual_duration > 0:
                    fps = (len(frames_in_window) - 1) / actual_duration
                    frame_rates.append(fps)

        if len(frame_rates) > 0:
            fps_mean = np.mean(frame_rates)
            fps_stddev = np.std(frame_rates)
            fps_min = np.min(frame_rates)
            fps_max = np.max(frame_rates)
            fps_variation = fps_max - fps_min

            if (
                fps_variation
                > THRESHOLDS["video"]["frame_rate_variation_threshold_fps"]
            ):
                issues.append("frame_rate_inconsistency")
                issue_details["frame_rate_inconsistency"] = {
                    "mean_fps": float(fps_mean),
                    "stddev_fps": float(fps_stddev),
                    "min_fps": float(fps_min),
                    "max_fps": float(fps_max),
                    "variation_fps": float(fps_variation),
                    "expected_fps": float(ref_fps),
                    "threshold_variation_fps": THRESHOLDS["video"][
                        "frame_rate_variation_threshold_fps"
                    ],
                    "description": "Frame rate varies significantly over time indicating VRR or encoding issues",
                }

    # Separate into errors and warnings
    warning_types = {"frame_drops", "vft_read_errors"}
    errors = [issue for issue in issues if issue not in warning_types]
    warnings = [issue for issue in issues if issue in warning_types]

    error_details = {k: v for k, v in issue_details.items() if k not in warning_types}
    warning_details = {k: v for k, v in issue_details.items() if k in warning_types}

    # Generate summary
    if len(errors) == 0 and len(warnings) == 0:
        summary = "ok"
    else:
        summary = errors + warnings

    return {
        "summary": summary,
        "frame_info": {
            "total_frames": int(total_frames),
            "valid_frames": int(valid_count),
            "expected_fps": float(ref_fps),
            "frame_time_ms": float(frame_time_ms),
        },
        "errors": error_details,
        "warnings": warning_details,
    }


def avsync_stats_function(
    avsync_results,
    output_stats,
    source_file=None,
    ref_fps=30.0,
    audio_results=None,
    beep_period_sec=3.0,
):
    """
    Calculate and write A/V sync statistics to JSON file.

    Args:
        avsync_results: DataFrame containing A/V sync analysis results with columns:
                        'avsync_sec', 'video_timestamp', 'audio_timestamp',
                        'video_value_read', 'video_value_read_smoothed'
        output_stats: Path to output JSON file for statistics
        source_file: Optional source filename for reference in JSON output
        ref_fps: Reference frame rate (default 30.0) for quantization-aware thresholds
        audio_results: Optional DataFrame containing audio beep detection results
        beep_period_sec: Expected spacing between audio beeps (default 3.0)
    """
    if avsync_results is None or len(avsync_results) == 0:
        print("No avsync results to write statistics")
        return

    # Calculate frame time for quantization-aware thresholds
    # Video is discrete, so there's inherent ±0.5 frame quantization error
    frame_time_ms = (1000.0 / ref_fps) if ref_fps > 0 else 33.33  # Default to 30fps

    # Calculate basic statistics
    avsync_sec_average = np.average(avsync_results["avsync_sec"])
    avsync_sec_stddev = np.std(avsync_results["avsync_sec"])
    avsync_sec_p50 = np.percentile(avsync_results["avsync_sec"], 50)
    avsync_sec_p90 = np.percentile(avsync_results["avsync_sec"], 90)
    avsync_sec_min = np.min(avsync_results["avsync_sec"])
    avsync_sec_max = np.max(avsync_results["avsync_sec"])

    # Convert to milliseconds for threshold comparisons
    avsync_ms = avsync_results["avsync_sec"] * 1000

    # Issue detection
    issues = []
    issue_details = {}

    # Use global thresholds and calculate frame-rate aware values
    # Note: Video quantization introduces ±0.5 frame time error inherently
    thresholds = {
        # Frame-rate independent thresholds from global config
        "itu_r_detectable_min_ms": THRESHOLDS["avsync"]["itu_r_detectable_min_ms"],
        "itu_r_detectable_max_ms": THRESHOLDS["avsync"]["itu_r_detectable_max_ms"],
        "itu_r_acceptable_min_ms": THRESHOLDS["avsync"]["itu_r_acceptable_min_ms"],
        "itu_r_acceptable_max_ms": THRESHOLDS["avsync"]["itu_r_acceptable_max_ms"],
        "drift_rate_threshold_ms_per_sec": THRESHOLDS["avsync"][
            "drift_rate_threshold_ms_per_sec"
        ],
        "drift_total_threshold_ms": THRESHOLDS["avsync"]["drift_total_threshold_ms"],
        "oscillation_correlation_threshold": THRESHOLDS["avsync"][
            "oscillation_correlation_threshold"
        ],
        "oscillation_min_period": THRESHOLDS["avsync"]["oscillation_min_period"],
        "drift_direction_min_slope": THRESHOLDS["avsync"]["drift_direction_min_slope"],
        "outlier_iqr_multiplier": THRESHOLDS["avsync"]["outlier_iqr_multiplier"],
        "outlier_percentage_threshold": THRESHOLDS["avsync"][
            "outlier_percentage_threshold"
        ],
        # Frame-rate aware thresholds (calculated from base values)
        "jitter_threshold_ms": max(
            THRESHOLDS["avsync"]["jitter_base_threshold_ms"], frame_time_ms
        ),
        "sudden_jump_threshold_ms": frame_time_ms
        * THRESHOLDS["avsync"]["sudden_jump_frame_multiplier"],
        "temporal_avg_change_threshold_ms": max(
            THRESHOLDS["avsync"]["temporal_avg_change_base_ms"],
            frame_time_ms
            * THRESHOLDS["avsync"]["temporal_avg_change_frame_multiplier"],
        ),
        "temporal_std_change_threshold_ms": max(
            THRESHOLDS["avsync"]["temporal_std_change_base_ms"],
            frame_time_ms
            * THRESHOLDS["avsync"]["temporal_std_change_frame_multiplier"],
        ),
    }

    # 1. ITU-R BT.1359-1 threshold violations
    detectable_violations = (
        (avsync_ms < thresholds["itu_r_detectable_min_ms"])
        | (avsync_ms > thresholds["itu_r_detectable_max_ms"])
    ).sum()
    acceptable_violations = (
        (avsync_ms < thresholds["itu_r_acceptable_min_ms"])
        | (avsync_ms > thresholds["itu_r_acceptable_max_ms"])
    ).sum()

    if acceptable_violations > 0:
        issues.append("itu_r_acceptable_threshold_violation")
        issue_details["itu_r_acceptable_threshold_violation"] = {
            "count": int(acceptable_violations),
            "percentage": float(acceptable_violations / len(avsync_results) * 100),
            "threshold_range_ms": [
                thresholds["itu_r_acceptable_min_ms"],
                thresholds["itu_r_acceptable_max_ms"],
            ],
            "actual_min_ms": float(avsync_sec_min * 1000),
            "actual_max_ms": float(avsync_sec_max * 1000),
            "description": "A/V sync values outside ITU-R BT.1359-1 acceptable range (-185ms to +90ms)",
        }
    elif detectable_violations > 0:
        issues.append("itu_r_detectable_threshold_violation")
        issue_details["itu_r_detectable_threshold_violation"] = {
            "count": int(detectable_violations),
            "percentage": float(detectable_violations / len(avsync_results) * 100),
            "threshold_range_ms": [
                thresholds["itu_r_detectable_min_ms"],
                thresholds["itu_r_detectable_max_ms"],
            ],
            "actual_min_ms": float(avsync_sec_min * 1000),
            "actual_max_ms": float(avsync_sec_max * 1000),
            "description": "A/V sync values outside ITU-R BT.1359-1 detectable range (-125ms to +45ms)",
        }

    # 2. Drift detection - calculate linear regression
    if "video_timestamp" in avsync_results.columns and len(avsync_results) > 1:
        timestamps = avsync_results["video_timestamp"].values
        avsync_values = avsync_results["avsync_sec"].values

        # Simple linear fit: avsync = drift_rate * time + offset
        coeffs = np.polyfit(timestamps, avsync_values, 1)
        drift_rate_sec_per_sec = coeffs[0]
        drift_rate_ms_per_sec = drift_rate_sec_per_sec * 1000

        # Calculate total drift over the measurement period
        time_span = timestamps[-1] - timestamps[0]
        total_drift_ms = drift_rate_ms_per_sec * time_span

        if (
            abs(drift_rate_ms_per_sec) > thresholds["drift_rate_threshold_ms_per_sec"]
            or abs(total_drift_ms) > thresholds["drift_total_threshold_ms"]
        ):
            issues.append("drift")
            issue_details["drift"] = {
                "drift_rate_ms_per_sec": float(drift_rate_ms_per_sec),
                "total_drift_ms": float(total_drift_ms),
                "time_span_sec": float(time_span),
                "actual_min_ms": float(avsync_sec_min * 1000),
                "actual_max_ms": float(avsync_sec_max * 1000),
                "threshold_rate_ms_per_sec": thresholds[
                    "drift_rate_threshold_ms_per_sec"
                ],
                "threshold_total_ms": thresholds["drift_total_threshold_ms"],
                "description": "Linear clock drift between audio and video over time",
            }

    # 3. High jitter/variance
    jitter_ms = avsync_sec_stddev * 1000

    if jitter_ms > thresholds["jitter_threshold_ms"]:
        issues.append("high_jitter")
        issue_details["high_jitter"] = {
            "stddev_ms": float(jitter_ms),
            "threshold_ms": thresholds["jitter_threshold_ms"],
            "description": "High variance in A/V sync measurements indicating unstable synchronization",
        }

    # 4. Sudden jumps/discontinuities
    if len(avsync_results) > 1:
        avsync_diff = np.diff(avsync_results["avsync_sec"].values)
        jump_threshold_sec = thresholds["sudden_jump_threshold_ms"] / 1000.0

        large_jumps = np.abs(avsync_diff) > jump_threshold_sec
        num_jumps = large_jumps.sum()

        if num_jumps > 0:
            jump_indices = np.where(large_jumps)[0]

            # Detect measurement errors: single bad points that cause jumps
            # For each jump, test if removing the landing point eliminates the issue
            measurement_errors = set()
            avsync_values = avsync_results["avsync_sec"].values

            for idx in jump_indices:
                # Test removing the landing point (idx+1)
                landing_idx = idx + 1
                if (
                    landing_idx < len(avsync_values) - 1
                ):  # Need at least one point after
                    # Create array without this point
                    test_values = np.concatenate(
                        [avsync_values[:landing_idx], avsync_values[landing_idx + 1 :]]
                    )
                    test_diff = np.diff(test_values)

                    # Check if jumps around this position disappear
                    # The jump at idx maps to idx in test_diff (before removal point)
                    # The jump at idx+1 maps to idx in test_diff (after removal point)
                    test_large_jumps = np.abs(test_diff) > jump_threshold_sec

                    # If removing this point eliminates the jump at this position
                    if idx < len(test_diff) and not test_large_jumps[idx]:
                        measurement_errors.add(landing_idx)

            # Filter out measurement errors from jump reporting
            valid_jumps = [
                idx for idx in jump_indices if (idx + 1) not in measurement_errors
            ]

            if len(valid_jumps) > 0:
                issues.append("sudden_jumps")

                # Get actual timestamps and frame numbers for jumps
                # np.diff index i represents transition from row i to row i+1
                # Report the landing point (i+1)
                jump_details = []
                for idx in valid_jumps[:10]:  # First 10 only
                    jump_info = {}
                    if "video_timestamp" in avsync_results.columns:
                        jump_info["video_timestamp_sec"] = float(
                            avsync_results.iloc[idx + 1]["video_timestamp"]
                        )
                    if "frame_num" in avsync_results.columns:
                        jump_info["frame_num"] = int(
                            avsync_results.iloc[idx + 1]["frame_num"]
                        )
                    jump_info["jump_magnitude_ms"] = float(abs(avsync_diff[idx]) * 1000)
                    jump_details.append(jump_info)

                issue_details["sudden_jumps"] = {
                    "count": len(valid_jumps),
                    "max_jump_ms": float(
                        np.max(np.abs(avsync_diff[valid_jumps])) * 1000
                    ),
                    "jump_locations": jump_details,
                    "threshold_ms": thresholds["sudden_jump_threshold_ms"],
                    "frame_time_ms": frame_time_ms,
                    "description": "Discontinuous jumps in A/V sync greater than 2 frame times",
                }

            # Report measurement errors separately if found
            if len(measurement_errors) > 0:
                issues.append("measurement_errors")
                error_details = []
                for err_idx in sorted(measurement_errors)[:10]:  # First 10 only
                    error_info = {}
                    if "video_timestamp" in avsync_results.columns:
                        error_info["video_timestamp_sec"] = float(
                            avsync_results.iloc[err_idx]["video_timestamp"]
                        )
                    if "frame_num" in avsync_results.columns:
                        error_info["frame_num"] = int(
                            avsync_results.iloc[err_idx]["frame_num"]
                        )
                    error_info["avsync_ms"] = float(
                        avsync_results.iloc[err_idx]["avsync_sec"] * 1000
                    )
                    error_details.append(error_info)

                issue_details["measurement_errors"] = {
                    "count": len(measurement_errors),
                    "error_locations": error_details,
                    "description": "Single bad measurements causing apparent jumps down and back up",
                }

    # 5. Periodic oscillation detection
    if len(avsync_results) > 10:
        # Detrend the signal first
        from scipy import signal

        avsync_detrended = signal.detrend(avsync_results["avsync_sec"].values)

        # Use autocorrelation to detect periodicity
        autocorr = np.correlate(avsync_detrended, avsync_detrended, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]  # Normalize

        # Look for peaks in autocorrelation (excluding the zero-lag peak)
        if len(autocorr) > 5:
            # Find local maxima in first half of autocorrelation
            search_range = min(len(autocorr) // 2, 50)
            local_max_indices = []
            for i in range(thresholds["oscillation_min_period"], search_range):
                if (
                    autocorr[i] > autocorr[i - 1]
                    and autocorr[i] > autocorr[i + 1]
                    and autocorr[i] > thresholds["oscillation_correlation_threshold"]
                ):
                    local_max_indices.append(i)

            if len(local_max_indices) > 0:
                issues.append("periodic_oscillation")
                issue_details["periodic_oscillation"] = {
                    "peak_periods_samples": local_max_indices[:5],  # First 5 periods
                    "autocorrelation_strength": float(autocorr[local_max_indices[0]]),
                    "threshold_correlation": thresholds[
                        "oscillation_correlation_threshold"
                    ],
                    "min_period_samples": thresholds["oscillation_min_period"],
                    "description": "Regular periodic pattern in A/V sync drift",
                }

    # 6. Non-monotonic drift
    if len(avsync_results) > 10:
        # Use moving window to detect drift direction changes
        window_size = max(5, len(avsync_results) // 10)
        drift_directions = []

        for i in range(0, len(avsync_results) - window_size, window_size):
            window = avsync_results["avsync_sec"].values[i : i + window_size]
            # Linear regression on window
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            # Only count as a direction if slope is significant
            if abs(slope) > thresholds["drift_direction_min_slope"]:
                drift_directions.append(1 if slope > 0 else -1)
            else:
                drift_directions.append(0)  # No significant drift

        # Count direction changes (ignore transitions to/from 0)
        if len(drift_directions) > 1:
            direction_changes = 0
            prev_nonzero = None
            for d in drift_directions:
                if d != 0:
                    if prev_nonzero is not None and d != prev_nonzero:
                        direction_changes += 1
                    prev_nonzero = d

            if direction_changes > 0:
                issues.append("non_monotonic_drift")
                issue_details["non_monotonic_drift"] = {
                    "direction_changes": direction_changes,
                    "num_segments": len(drift_directions),
                    "min_slope_threshold": thresholds["drift_direction_min_slope"],
                    "description": "Drift direction changes over time (audio catching up then falling behind)",
                }

    # 7. Temporal segmentation - compare first and last thirds
    if len(avsync_results) > 30:
        segment_size = len(avsync_results) // 3
        first_third = avsync_results["avsync_sec"].values[:segment_size]
        last_third = avsync_results["avsync_sec"].values[-segment_size:]

        first_avg = np.mean(first_third)
        last_avg = np.mean(last_third)
        first_std = np.std(first_third)
        last_std = np.std(last_third)

        avg_change_ms = abs(last_avg - first_avg) * 1000
        std_change_ms = abs(last_std - first_std) * 1000

        if (
            avg_change_ms > thresholds["temporal_avg_change_threshold_ms"]
            or std_change_ms > thresholds["temporal_std_change_threshold_ms"]
        ):
            issues.append("temporal_segmentation")
            issue_details["temporal_segmentation"] = {
                "first_third_avg_ms": float(first_avg * 1000),
                "last_third_avg_ms": float(last_avg * 1000),
                "avg_change_ms": float(avg_change_ms),
                "first_third_std_ms": float(first_std * 1000),
                "last_third_std_ms": float(last_std * 1000),
                "std_change_ms": float(std_change_ms),
                "threshold_avg_change_ms": thresholds[
                    "temporal_avg_change_threshold_ms"
                ],
                "threshold_std_change_ms": thresholds[
                    "temporal_std_change_threshold_ms"
                ],
                "description": "Significant change in A/V sync behavior between beginning and end of recording",
            }

    # 8. Outlier detection using IQR method
    q1 = np.percentile(avsync_results["avsync_sec"], 25)
    q3 = np.percentile(avsync_results["avsync_sec"], 75)
    iqr = q3 - q1

    lower_bound = q1 - thresholds["outlier_iqr_multiplier"] * iqr
    upper_bound = q3 + thresholds["outlier_iqr_multiplier"] * iqr

    outliers = (avsync_results["avsync_sec"] < lower_bound) | (
        avsync_results["avsync_sec"] > upper_bound
    )
    num_outliers = outliers.sum()
    outlier_percentage = num_outliers / len(avsync_results) * 100

    if outlier_percentage > thresholds["outlier_percentage_threshold"]:
        issues.append("outliers")
        issue_details["outliers"] = {
            "count": int(num_outliers),
            "percentage": float(outlier_percentage),
            "iqr_bounds_ms": [float(lower_bound * 1000), float(upper_bound * 1000)],
            "iqr_multiplier": thresholds["outlier_iqr_multiplier"],
            "percentage_threshold": thresholds["outlier_percentage_threshold"],
            "description": "Statistical outliers in A/V sync measurements using IQR method",
        }

    # Separate issues into errors and warnings
    warning_types = {"measurement_errors", "itu_r_detectable_threshold_violation"}

    errors = [issue for issue in issues if issue not in warning_types]
    warnings = [issue for issue in issues if issue in warning_types]

    error_details = {k: v for k, v in issue_details.items() if k not in warning_types}
    warning_details = {k: v for k, v in issue_details.items() if k in warning_types}

    # Generate summary
    if len(errors) == 0 and len(warnings) == 0:
        summary = "ok"
    else:
        summary = errors + warnings

    # Write statistics to JSON file if output_stats is provided
    if output_stats:
        stats = {
            "avsync": {
                "summary": summary,
                "avsync_sec": {
                    "average": float(avsync_sec_average),
                    "stddev": float(avsync_sec_stddev),
                    "min": float(avsync_sec_min),
                    "max": float(avsync_sec_max),
                    "p50": float(avsync_sec_p50),
                    "p90": float(avsync_sec_p90),
                    "size": len(avsync_results),
                },
                "errors": error_details,
                "warnings": warning_details,
            }
        }

        # Add audio statistics if audio_results is provided
        if audio_results is not None and len(audio_results) > 0:
            # Calculate total duration from audio timestamps
            total_duration_sec = (
                audio_results["timestamp"].max() if len(audio_results) > 0 else None
            )

            audio_stats = audio_stats_function(
                audio_results,
                output_stats,
                source_file,
                beep_period_sec,
                total_duration_sec,
            )

            if audio_stats:
                stats["audio"] = audio_stats

        # Add video statistics if we have avsync results
        if len(avsync_results) > 0:
            video_stats = video_stats_function(
                avsync_results,
                output_stats,
                source_file,
                ref_fps,
            )

            if video_stats:
                stats["video"] = video_stats

        # Add source file as first entry if provided
        if source_file:
            stats = {"file": source_file, **stats}

        # Add metiq version and command information
        metiq_info = {}

        # Get git version
        try:
            git_version = subprocess.check_output(
                ["/usr/bin/git", "describe", "HEAD"],
                cwd=os.path.dirname(__file__),
                text=True,
            ).strip()
            metiq_info["version"] = git_version
        except (subprocess.CalledProcessError, FileNotFoundError):
            metiq_info["version"] = "unknown"

        # Get command line
        metiq_info["command"] = " ".join(sys.argv)

        # Add metiq section to stats
        stats = {"metiq": metiq_info, **stats}

        with open(output_stats, "w") as f:
            json.dump(stats, f, indent=2)


def avsync_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    ref_fps = kwargs.get("ref_fps")
    beep_period_sec = kwargs.get("beep_period_sec")
    debug = kwargs.get("debug")
    z_filter = kwargs.get("z_filter")
    outfile = kwargs.get("outfile")
    output_stats = kwargs.get("output_stats")
    video_smoothed = kwargs.get("video_smoothed", False)

    # av sync is the time from the signal until the video is shown
    # for tests that include a transmission the signal of interest is
    # the first echo and not the source.

    if len(audio_results) == 0:
        print("No audio results, skipping av sync calculation")
        return

    # Always apply smoothing for avsync to correct VFT reading errors
    video_results = calculate_value_read_smoothed(video_results, ref_fps=ref_fps)

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "avsync")

    margin = 0.7
    clean_audio = filter_echoes(audio_results, beep_period_sec, margin)
    # Check residue
    signal_ratio = len(clean_audio) / len(audio_results)
    if signal_ratio < 1:
        print(
            f"\nRemoved {signal_ratio * 100:.2f}% echoes, transmission use case. Video latency can be calculated.\n"
        )
        if signal_ratio < 0.2:
            print("Few echoes, recheck thresholds")

        # Filter residues to get echoes
        residue = audio_results[~audio_results.index.isin(clean_audio.index)]
        clean_audio = filter_echoes(pd.DataFrame(residue), beep_period_sec, margin)

    else:
        print(
            "\nWarning, no echoes, simple source use case. No video latency calculation possible.\n"
        )

    avsync_results = calculate_avsync(
        clean_audio,
        video_results,
        fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
    )
    # filter the a/v sync values
    if z_filter > 0:
        avsync_results = z_filter_function(avsync_results, "avsync_sec", z_filter)
    if len(avsync_results) > 0:
        avsync_results.to_csv(outfile, index=False)

    # calculate and write statistics
    if output_stats is not None:
        # Extract source filename from input_video
        input_video = kwargs.get("input_video", None)
        source_file = None
        if input_video:
            # Remove path and .video.csv extension
            base = os.path.basename(input_video)
            if base.endswith(".video.csv"):
                source_file = base[: -len(".video.csv")]
            else:
                source_file = base

        avsync_stats_function(
            avsync_results,
            output_stats,
            source_file,
            ref_fps,
            audio_results,
            beep_period_sec,
        )


def calculate_video_playouts(video_results):
    # 1 add value_read_delta (difference between the value read between consecutive frames)
    video_results["value_read_delta"] = [
        0,
    ] + list(
        y - x
        for (x, y) in zip(video_results.value_read[:-1], video_results.value_read[1:])
    )
    # 2. remove consecutive frames with the same value read
    video_results = video_results.drop(
        video_results[video_results.value_read_delta == 0].index
    )
    # 3. add value_read_delta_minus_mean (column assuming the average delta)
    average_delta = round(video_results.value_read_delta.mean())
    video_results["value_read_delta_minus_mean"] = (
        video_results.value_read_delta - average_delta
    )
    # 4. remove unused columns
    # TODO: or keep wanted?
    unused_col_names = (
        "frame_num_expected",
        "delta_frame",
        "value_before_clean",
        "value_read_delta",
    )
    unused_col_names = [
        col for col in unused_col_names if col in video_results.columns.values
    ]
    video_results = video_results.drop(
        columns=unused_col_names,
        axis=1,
    )
    return video_results


def filter_halfsteps(video_results):
    # Halfsteps are the result of the video signal being read in between frames
    # We cannot know what it really should be. Let us do the following:
    # If the value is .5 from previous value, use previous value.
    # If the value is .5 from next value, use next value.
    # else use round up (time moves forward most of the time).
    video_results = pd.DataFrame(video_results)
    half_values = video_results.loc[video_results["value_read"].mod(1) == 0.5]
    if len(half_values) == 0:
        return video_results
    for index, row in half_values.iterrows():
        if index == 0 or index == len(video_results) - 1:
            continue
        if abs(row["value_read"] - video_results.at[index - 1, "value_read"]) == 0.5:
            video_results.at[index, "value_read"] = video_results.at[
                index - 1, "value_read"
            ]
        elif abs(row["value_read"] - video_results.at[index + 1, "value_read"]) == 0.5:
            video_results.at[index, "value_read"] = video_results.at[
                index + 1, "value_read"
            ]
        else:
            video_results.at[index, "value_read"] = math.floor(row["value_read"])
    return video_results


def filter_ambiguous_framenumber(video_results):
    video_results = filter_halfsteps(video_results)
    # one frame cannot have a different value than two adjacent frames.
    # this is only true if the capture fps is at least twice the draw frame rate (i.e. 240fps at 120Hz display).
    # Use next value

    # no holes please
    video_results["value_read"].ffill(inplace=True)
    # Maybe some values in the beginning are bad as well.
    video_results["value_read"].bfill(inplace=True)
    video_results["value_clean"] = video_results["value_read"].astype(int)
    video_results["val_m1"] = video_results["value_clean"].shift(-1)
    video_results["val_p1"] = video_results["value_clean"].shift(1)
    video_results["val_m1"].ffill(inplace=True)
    video_results["val_p1"].bfill(inplace=True)

    video_results["singles"] = (
        video_results["value_clean"] != video_results["val_m1"]
    ) & (video_results["value_clean"] != video_results["val_p1"])
    video_results.loc[video_results["singles"], "value_clean"] = np.NaN
    video_results["value_clean"].ffill(inplace=True)
    video_results["value_clean"] = video_results["value_clean"].astype(int)
    video_results["value_before_clean"] = video_results["value_read"]

    # use the new values in subsequent analysis
    video_results["value_read"] = video_results["value_clean"]
    video_results.drop(
        columns=["val_m1", "val_p1", "singles", "value_clean"], inplace=True
    )
    return video_results


def quality_stats_function(**kwargs):
    audio_results = kwargs.get("audio_results")
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "quality_stats")

    quality_stats_results = calculate_measurement_quality_stats(
        audio_results, video_results
    )
    quality_stats_results.to_csv(outfile, index=False)


def windowed_stats_function(**kwargs):
    video_results = kwargs.get("video_results")
    windowed_stats_sec = kwargs.get("windowed_stats_sec")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "windowed_stats")

    windowed_stats_results = calculate_frames_moving_average(
        video_results, windowed_stats_sec
    )
    if len(windowed_stats_results) > 0:
        windowed_stats_results.to_csv(outfile, index=False)
    else:
        print("No windowed stats to write")


def frame_duration_function(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "frame_duration")

    frame_duration_results = calculate_frame_durations(video_results)
    if len(frame_duration_results) > 0:
        frame_duration_results.to_csv(outfile, index=False)
    else:
        print("No frame durations to write")


def video_playout_function(**kwargs):
    video_results = kwargs.get("video_results")
    outfile = kwargs.get("outfile")

    if not outfile:
        infile = kwargs.get("input_video", None)
        outfile = create_output_filename(infile, "video_playout")

    video_playout_results = calculate_video_playouts(video_results)
    if len(video_playout_results) > 0:
        video_playout_results.to_csv(outfile, index=False)
    else:
        print("No video playouts to write")


def media_analyze(
    analysis_type,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    input_video,
    input_audio,
    outfile,
    output_stats,
    force_fps,
    audio_offset,
    filter_all_echoes,
    z_filter,
    windowed_stats_sec,
    cleanup_video=False,
    video_smoothed=True,
    min_match_threshold=None,
    debug=0,
):
    # read inputs
    video_results = None
    try:
        video_results = pd.read_csv(input_video)
        # Remove obvious errors
        if cleanup_video:
            video_results = filter_ambiguous_framenumber(video_results)
    except ValueError:
        # ignore in case the analysis does not need it
        if debug > 0:
            print("No video data")
        pass
    audio_results = None
    try:
        audio_results = pd.read_csv(input_audio)
    except ValueError:
        # ignore in case the analysis does not need it
        pass

    # filter audio thresholds
    if audio_results is not None and min_match_threshold is not None:
        audio_results = audio_results.loc[
            audio_results["correlation"] >= min_match_threshold
        ]
    # estimate the video framerate
    # TODO: capture fps should be available
    ref_fps, capture_fps = video_parse.estimate_fps(video_results)
    if force_fps > 0:
        ref_fps = force_fps

    # adjust the audio offset
    if audio_offset is not None:
        video_results["timestamp"] += audio_offset

    if filter_all_echoes:
        audio_results = filter_echoes(audio_results, beep_period_sec, 0.7)

    assert analysis_type is not None, f"error: need to specify --analysis-type"
    analysis_function = MEDIA_ANALYSIS[analysis_type][0]
    analysis_function(
        audio_results=audio_results,
        video_results=video_results,
        fps=ref_fps,  # TODO(chema): only one
        ref_fps=ref_fps,
        beep_period_sec=beep_period_sec,
        debug=debug,
        outfile=outfile,
        output_stats=output_stats,
        z_filter=z_filter,
        windowed_stats_sec=windowed_stats_sec,
        input_video=input_video,
        video_smoothed=video_smoothed,
    )


MEDIA_ANALYSIS = {
    "audio_latency": (
        audio_latency_function,
        "Calculate audio latency",
        ".audio.latency.csv",
    ),
    "video_latency": (
        video_latency_function,
        "Calculate video latency",
        ".video.latency.csv",
    ),
    "avsync": (
        avsync_function,
        "Calculate audio/video synchronization offset using audio timestamps and video frame numbers",
        ".avsync.csv",
    ),
    "quality_stats": (
        quality_stats_function,
        "Calculate quality stats",
        ".measurement.quality.csv",
    ),
    "windowed_stats": (
        windowed_stats_function,
        "Calculate video frames shown/dropped per unit sec",
        ".windowed.stats.csv",
    ),
    "frame_duration": (
        frame_duration_function,
        "Calculate source frame durations",
        ".frame.duration.csv",
    ),
    "video_playout": (
        video_playout_function,
        "Analyze video playout issues",
        ".video.playout.csv",
    ),
    "all": (all_analysis_function, "Calculate all media analysis", None),
}
