import json
import random
from pathlib import Path
from logging import getLogger

import torch
from torch.utils import data

logger = getLogger()

SPLIT_SHARD_RANGES = {
    "train": (0, 1200),
    "valid": (1200, 1600),
    "test": (1600, 2000),
}


class HCPYA_sex_scale(data.Dataset):
    def __init__(
        self,
        split: str,
        processed_dir: str,
        use_normalization=False,
        downsample=False,
        sampling_rate=3,
        num_frames=160,
        keep_in_memory=True,
    ):
        self.split = split
        self.processed_dir = processed_dir
        self.use_normalization = use_normalization

        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        self.keep_in_memory = keep_in_memory

        self.n_rois = 450
        self.seq_length = 490

        start, stop = SPLIT_SHARD_RANGES[split]
        self.sample_paths = [
            str(path)
            for shard in range(start, stop)
            for path in sorted(
                (Path(processed_dir) / f"hcp-parc_{shard:04d}").glob("*_task-REST*_mag-3T_*.pt")
            )
        ]
        if keep_in_memory:
            self.samples = [
                torch.load(path, map_location="cpu", weights_only=True)
                for path in self.sample_paths
            ]

        with (Path(processed_dir) / "hcp_sex_target_id_map.json").open() as f:
            self.target_map = json.load(f)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if self.keep_in_memory:
            sample = self.samples[idx]
        else:
            path = self.sample_paths[idx]
            sample = torch.load(path, map_location="cpu", weights_only=True)

        input_x = sample["bold"] # [n_frames, n_rois]
        assert len(input_x) >= self.seq_length
        input_x = input_x[:self.seq_length].t().contiguous().float()  # [n_rois, n_frames]
        label_y = self.target_map[sample["sub"]]

        if self.use_normalization:
            mean = input_x.mean()
            std = input_x.std()
            input_x = (input_x - mean) / std

        if self.downsample:
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(self.seq_length, clip_size)
            ts_array = self._temporal_sampling(
                        input_x, start_idx, end_idx, self.num_frames
                    )
            input_x = torch.unsqueeze(ts_array, 0).to(torch.float32)
        else:
            input_x = torch.unsqueeze(input_x, 0).to(torch.float32)

        return input_x.to(torch.float32), int(label_y)

    def _get_start_end_idx(self, fmri_size, clip_size):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames


def make_hcya_sex(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    drop_last=True,
    processed_dir='data/hcp_parc',
    use_normalization=False,
    label_normalization=False,
    downsample=False,
):
    # train data loader
    train_dataset = HCPYA_sex_scale(
        split='train',
        processed_dir=processed_dir,
        use_normalization=use_normalization,
        downsample=downsample
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    # validation data loader
    valid_dataset = HCPYA_sex_scale(
        split='valid',
        processed_dir=processed_dir,
        use_normalization=use_normalization,
        downsample=downsample
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    # test data loader
    test_dataset = HCPYA_sex_scale(
        split='test',
        processed_dir=processed_dir,
        use_normalization=use_normalization,
        downsample=downsample)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info('hca_sex dataset created')

    return train_data_loader, valid_data_loader, test_data_loader, train_dataset, valid_dataset, test_dataset

