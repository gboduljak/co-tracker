import os
import json
import cv2
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from cotracker.datasets.tap_vid_datasets import sample_queries_first, sample_queries_strided
from cotracker.datasets.utils import CoTrackerData


def load_mp4_to_frames(filename: str, to_rgb: bool = True) -> np.ndarray:
    """Loads an MP4 video file and returns its frames as a NumPy array.

    Args:
        filename (str): Path to the MP4 video file.
        to_rgb (bool): Whether to convert frames from BGR (OpenCV default) to RGB.

    Returns:
        np.ndarray: Frames of the video as a NumPy array.
    """
    assert os.path.exists(filename), f'File {filename} does not exist.'
    cap = cv2.VideoCapture(filename)

    vid_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    vid_frames = np.empty((vid_frames_count, height, width, 3), dtype=np.uint8)

    idx = 0
    while True:
        ret, vid_frame = cap.read()
        if not ret:
            break
        if to_rgb:
            vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
        vid_frames[idx] = vid_frame
        idx += 1

    cap.release()
    return vid_frames


class PerceptionTest(Dataset):
    def __init__(self, dataset_root: str, split: str, queried_first: bool = True):
        """
        Args:
            dataset_root (str): Path to dataset root folder.
            split (str): 'train' or 'valid'.
        """
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        self.dataset_root = Path(dataset_root) / split
        self.queried_first = queried_first

        # Path to label file (you may need to adapt extension if it's not JSON)
        label_file = self.dataset_root / "point_tracking.json"
        assert label_file.exists(), f"Label file not found: {label_file}"

        with open(label_file, "r", encoding="utf-8") as fs:
            label_dict = json.load(fs)

        self.metadata = {}
        for video_id, gt in label_dict.items():
            try:
                gt_tracks = gt["point_tracking"]
                num_frames = gt["metadata"]["num_frames"]
                num_tracks = len(gt_tracks)

                query_points = np.zeros((num_tracks, 3), dtype=np.float32)
                gt_occluded = np.ones((num_tracks, num_frames), dtype=np.uint8)
                gt_points = np.zeros((num_tracks, num_frames, 2), dtype=np.float32)

                for track_meta in gt_tracks:
                    gt_track_idx = track_meta["id"]
                    gt_track_points = np.array(track_meta["points"]).T
                    start_point = gt_track_points[0]
                    start_frame_id = track_meta["frame_ids"][0]

                    query_points[gt_track_idx, 0] = start_frame_id
                    query_points[gt_track_idx, 1:] = start_point
                    gt_occluded[gt_track_idx][track_meta["frame_ids"]] = 0
                    gt_points[gt_track_idx][track_meta["frame_ids"]] = (
                        gt_track_points[..., [1, 0]]
                    )

                self.metadata[video_id] = {
                    "query_points": query_points,
                    "occluded": gt_occluded,
                    "tracks": gt_points,
                }
            except Exception:
                continue

        # Store list of video_ids for indexing
        self.video_ids = sorted(self.metadata.keys())
        print("found %d unique videos in %s" % (len(self), str(dataset_root)))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx) -> CoTrackerData:
        video_id = self.video_ids[idx]
        meta = self.metadata[video_id]
        frames_raw = load_mp4_to_frames(
            str(self.dataset_root / "videos" / f"{video_id}.mp4")
        )
        frames = torch.from_numpy(frames_raw).float() / 255.0 # [t, h, w, c]
        frames = frames.permute(0, 3, 1, 2)
        frames = (
            F.interpolate(
              frames,
              size=(256, 256),
              mode='bilinear',
              align_corners=False
            ) * 255
        )

        if self.queried_first:
            converted = sample_queries_first(meta["occluded"], meta["tracks"] * 256.0, frames_raw)
        else:
            converted = sample_queries_strided(meta["occluded"], meta["tracks"] * 256.0, frames_raw)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, D
        # rgbs = frames.permute(0, 3, 1, 2).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N

        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
        
        t, *_ = frames.shape

        return CoTrackerData(
            frames,
            trajs[:t, ...],
            visibles[:t, ...],
            seq_name=video_id,
            query_points=query_points,
        )
        # return CoTrackerData(
        #     video=frames,
        #     trajectory=rearrange(
        #         torch.from_numpy(meta["tracks"]),
        #         "n t d -> t n d"
        #     ).contiguous(),
        #     query_points=torch.from_numpy(meta["query_points"]).contiguous(),
        #     visibility=rearrange(
        #         ~torch.from_numpy(meta["occluded"]).byte(),
        #         "n t -> t n"
        #     ).contiguous(),
        #     seq_name=video_id
        # )
