#! /usr/bin/env python3


import unittest
import inspect
import os
import sys

script_dir = os.path.dirname(__file__)
metiq_path = f"{script_dir}/../src"
sys.path.append(metiq_path)
sys.path.append(script_dir)
import glob
import pandas as pd
import common
import verify_config as config
import verify_generate as vg


def run_metiq_cli(**settings):
    filename = settings.get("outfile", "")
    audio_offset = settings.get("audio_offset", 0)
    command = f"python3 {metiq_path}/metiq.py parse -i {filename} --lock-layout -d"
    ret, stdout, stderr = common.run(command, debug=config.DEBUG)
    assert ret == 0, f"error: {stderr}"

    command = f"python3 {metiq_path}/metiq.py analyze --input-audio {filename}.audio.csv --input-video {filename}.video.csv --audio-offset {audio_offset} -a all -d"
    ret, stdout, stderr = common.run(command, debug=config.DEBUG)
    assert ret == 0, f"error: {stderr}"


def verify_metiq_cli(**settings):
    global DEBUG
    failed = False
    doc = settings.get("doc", "")
    filename = settings.get("outfile")
    audio_offset = settings.get("audio_offset", 0.0)
    video_delay = settings.get("video_delay", 0)
    audio_delay = settings.get("audio_delay", 0)
    av_sync = round(video_delay - audio_delay, 2)

    print(f"\n{'-'*20}\n{filename}\n")
    print(f"{doc}")
    print("Audio delay: ", audio_delay)
    print("Video delay: ", video_delay)
    print("A/V sync (calculated): ", av_sync)
    print("Audio offset: ", audio_offset)
    # read the files and compare
    if video_delay > 0:
        videolat = pd.read_csv(f"{filename}.video.latency.csv")
        meanval = videolat["video_latency_sec"].mean()
        result = (
            meanval < config.PREC + video_delay and meanval > video_delay - config.PREC
        )
        if not result:
            failed = True
            print(f"Video delay measurement failed: video delay: {meanval}")

        audiolat = pd.read_csv(f"{filename}.audio.latency.csv")
        meanval = audiolat["audio_latency_sec"].mean()
        result = (
            meanval < config.PREC + audio_delay and meanval > audio_delay - config.PREC
        )
        if not result:
            failed = True
            print(f"Audio delay measurement failed: audio delay: {meanval}")

    avsync = pd.read_csv(f"{filename}.avsync.csv")
    meanval = avsync["avsync_sec"].mean()
    result = meanval < config.PREC + av_sync and meanval > av_sync - config.PREC
    if not result:
        failed = True
        print(f"Audio/video synchronization measurement failed: a/v sync: {meanval}")

    if failed:
        print(f"{filename}")
        print(f"!!! FAILED\n---\n")
        # Keep files if broken test
        if video_delay > 0:
            print(f"Video latency:\n{videolat}")
            print(f"Audio latency:\n{audiolat}")
        print(f"A/V sync:\n{avsync}")
        return False
    else:
        print(f"PASS\n{'-'*20}\n")
        # remove all test files
        if not config.KEEP_FILES:
            for file in glob.glob(f"{filename}*"):
                os.remove(file)

    return True


# ----------------------------------


class E2ETests(unittest.TestCase):

    def test_1(self):
        """
        Plain file with no delays and perfect a/v sync. File @ 60fps
        """
        audio = 0.0
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"

        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
        }

        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_2(self):
        """
        Audio late 30 ms
        """
        audio = 0.030
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
        }

        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_3(self):
        """
        Audio late 30 ms, compensate the delay with audio offset.
        """
        audio = 0.030
        video = 0.0
        offset = 0.030
        descr = f"{inspect.currentframe().f_code.co_name}"
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "audio_offset": offset,
        }

        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_4(self):
        """
        audio early 30 ms
        """
        audio = -0.030
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
        }

        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_5(self):
        """
        Video delay 100ms, a/v sync perfect
        """
        audio = 0.200
        video = 0.200
        offset = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"

        settings = {
            "outfile": f"{descr}.mov",
            "descr": "vd.100ms",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "audio_offset": offset,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_6(self):
        """
        Audio late 200ms, video delay 100ms
        """
        audio = 0.200
        video = 0.100
        descr = f"{inspect.currentframe().f_code.co_name}"
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
        }

    def test_7(self):
        """
        Black frames outside of sync position. No delays and perfect sync.
        """
        audio = 0.0
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"
        # In 60 fps number scheme
        black_frames = [*range(15, 20), *range(70, 74)]
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "black_frames": black_frames,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_8(self):
        """
        Black frames at sync position. No delays and perfect sync.
        """
        audio = 0.0
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"
        # In 60 fps number scheme
        frames = [*range(160, 190)]
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "black_frames": frames,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_9(self):
        """
        Black frames at sync position with 200 ms video delay. Pefect sync.
        """
        audio = 0.200
        video = 0.200
        descr = f"{inspect.currentframe().f_code.co_name}"
        # In 60 fps number scheme
        frames = [*range(160, 190)]
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "black_frames": frames,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_10(self):
        """
        Frozen frames at sync position. No delays and perfect sync.
        """
        audio = 0.0
        video = 0.0
        descr = f"{inspect.currentframe().f_code.co_name}"
        # In 60 fps number scheme
        frames = [*range(160, 190)]
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "freeze_frames": frames,
        }

        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_11(self):
        """
        Frozen frames at sync position with 500 ms video delay. Audio 300ms early.
        """
        audio = 0.200
        video = 0.500
        descr = f"{inspect.currentframe().f_code.co_name}"
        # In 60 fps number scheme
        frames = [*range(160, 190)]
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "freeze_frames": frames,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)

    def test_12(self):
        """
        Offset is close to the video delay.
        """
        audio = 0.5
        video = 0.5
        offset = 0.5
        descr = f"{inspect.currentframe().f_code.co_name}"
        settings = {
            "outfile": f"{descr}.mov",
            "descr": f"{descr}",
            "output_fps": 60,
            "video_delay": video,
            "audio_delay": audio,
            "audio_offset": offset,
        }
        vg.generate_test_file(**settings)
        run_metiq_cli(**settings)

        verify_metiq_cli(**settings)


if __name__ == "__main__":
    unittest.main()
