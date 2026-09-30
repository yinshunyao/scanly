"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.
"""

import sys
import time
import pickle
import datetime
from collections import defaultdict, deque
from typing import Dict, Optional

import torch
import torch.distributed as tdist

from .dist_utils import is_dist_available_and_initialized, get_world_size

_MAIN_METERS = (
    ("lr", "lr"),
    ("loss", "loss"),
    ("loss_vfl", "vfl"),
    ("loss_bbox", "bbox"),
    ("loss_giou", "giou"),
)
_GROUP_METERS = (("aux", "_aux_"), ("dn", "_dn_"), ("enc", "_enc_"))
_TRAIN_COLS = (
    ("Epoch", 8),
    ("GPU_mem", 8),
    ("lr", 9),
    ("loss", 7),
    ("vfl", 7),
    ("bbox", 7),
    ("giou", 7),
    ("aux", 7),
    ("dn", 7),
    ("enc", 7),
)
_printed_train_header = False


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_available_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    tdist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    tdist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True) -> Dict[str, torch.Tensor]:
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 10) -> str:
    frac = min(1.0, max(0.0, frac))
    filled = int(round(frac * width))
    filled = min(width, max(0, filled))
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if enc.startswith("utf"):
        return "█" * filled + " " * (width - filled)
    return "#" * filled + "-" * (width - filled)


def _gpu_mem() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_reserved() / (1024 ** 3):.3g}G"
    return "-"


def _fmt_cell(value: Optional[float], width: int, *, kind: str) -> str:
    if value is None:
        return f"{'-':>{width}}"
    if kind == "e":
        return f"{value:>{width}.2e}"
    return f"{value:>{width}.4g}"


def train_header_line() -> str:
    return " ".join(f"{name:>{width}}" for name, width in _TRAIN_COLS)


def format_progress_bar(cur: int, total: int, elapsed: float, it_avg: float) -> str:
    frac = (cur / total) if total else 1.0
    pct = 100.0 * frac
    remain = max(0.0, it_avg * max(0, total - cur))
    rate = f"{(1.0 / it_avg):.2f}it/s" if it_avg > 0 else "?it/s"
    return (
        f"{pct:3.0f}%|{_bar(frac)}| {cur}/{total} "
        f"[{_hms(elapsed)}<{_hms(remain)}, {rate}]"
    )


class _LiveLine:
    """YOLO tqdm：100% 前只 \\r，结束才换行。"""

    def __init__(self):
        self._len = 0

    def emit(self, text: str, *, finish: bool) -> None:
        pad = max(0, self._len - len(text))
        print(text + (" " * pad), end="\n" if finish else "\r", flush=True)
        self._len = 0 if finish else len(text)


class MetricLogger(object):
    def __init__(self, delimiter="\t", compact=False, phase=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.compact = compact
        self.phase = phase or ("train" if compact else None)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def compact_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for src, label in _MAIN_METERS:
            meter = self.meters.get(src)
            if meter is None or meter.count <= 0:
                continue
            out[label] = meter.value if label == "lr" else meter.global_avg
        for label, token in _GROUP_METERS:
            names = [n for n in self.meters if token in n and self.meters[n].count > 0]
            if not names:
                continue
            out[label] = sum(self.meters[n].global_avg for n in names)
        return out

    def __str__(self):
        if self.compact:
            return self._compact_str()
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def _compact_str(self):
        """主头 + aux/dn/enc 合计，单值（epoch running mean）。"""
        values = self.compact_values()
        parts = []
        for _, label in _MAIN_METERS:
            if label not in values:
                continue
            spec = ".2e" if label == "lr" else ".4g"
            parts.append(f"{label}: {values[label]:{spec}}")
        for label, _token in _GROUP_METERS:
            if label not in values:
                continue
            parts.append(f"{label}: {values[label]:.4g}")
        return self.delimiter.join(parts)

    def _format_train_row(self, epoch_label: str) -> str:
        values = self.compact_values()
        cells = {
            "Epoch": f"{epoch_label:>8}",
            "GPU_mem": f"{_gpu_mem():>8}",
            "lr": _fmt_cell(values.get("lr"), 9, kind="e"),
            "loss": _fmt_cell(values.get("loss"), 7, kind="g"),
            "vfl": _fmt_cell(values.get("vfl"), 7, kind="g"),
            "bbox": _fmt_cell(values.get("bbox"), 7, kind="g"),
            "giou": _fmt_cell(values.get("giou"), 7, kind="g"),
            "aux": _fmt_cell(values.get("aux"), 7, kind="g"),
            "dn": _fmt_cell(values.get("dn"), 7, kind="g"),
            "enc": _fmt_cell(values.get("enc"), 7, kind="g"),
        }
        return " ".join(cells[name] for name, _width in _TRAIN_COLS)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if self.compact:
            yield from self._log_every_compact(iterable, print_freq, header)
            return
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def _log_every_compact(self, iterable, print_freq, header=None):
        global _printed_train_header
        header = header or ""
        total = len(iterable)
        line = _LiveLine()
        if self.phase == "train" and not _printed_train_header:
            print(train_header_line(), flush=True)
            _printed_train_header = True
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        i = 0
        try:
            for obj in iterable:
                yield obj
                iter_time.update(time.time() - end)
                last = i == total - 1
                bar = format_progress_bar(
                    i + 1, total, time.time() - start_time, iter_time.global_avg
                )
                if self.phase == "train":
                    text = f"{self._format_train_row(header)}: {bar}"
                else:
                    text = f"{header}: {bar}"
                line.emit(text, finish=last)
                i += 1
                end = time.time()
        finally:
            if line._len:
                print(flush=True)

