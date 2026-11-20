"""Process-level torch setup shared by the training scripts."""

from __future__ import annotations

import torch


def configure_torch(seed: int) -> None:
  """Seed torch and allow TF32 matmuls for the MLPs (small rotation products
  are written element-wise and stay FP32)."""
  torch.manual_seed(seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True


def limit_gpu_memory(device: str, gigabytes: float) -> None:
  """Cap this process's GPU allocations so it cannot crowd out other users of
  the card; exceeding the cap raises an out-of-memory error here instead."""
  if not device.startswith("cuda") or gigabytes <= 0:
    return
  index = torch.device(device).index or 0
  total = torch.cuda.get_device_properties(index).total_memory
  torch.cuda.set_per_process_memory_fraction(
    min(1.0, gigabytes * 1024**3 / total), index
  )


def gpu_memory_gb(device: str) -> float:
  """Peak GPU memory this process has reserved, in GiB (0 on the CPU)."""
  if not device.startswith("cuda"):
    return 0.0
  return torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
