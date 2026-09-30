"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import os
import time
import json
import datetime

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

# COCO bbox.stats 默认 12 项：0=AP@0.50:0.95，1=AP@0.50，8=AR@maxDets=100
_COCO_AP = 0
_COCO_AP50 = 1
_COCO_AR100 = 8
# 选 best.pth 的 val 指标；检测无 mask，canonical 为 ap50/ap/ar/yolo
_VAL_FITNESS_ALIASES = {
    "ap50": "ap50",
    "box_map50": "ap50",
    "map50": "ap50",
    "ap": "ap",
    "box_map": "ap",
    "map": "ap",
    "default": "ap",
    "ar": "ar",
    "recall": "ar",
    "yolo": "yolo",
}
_VAL_FITNESS_DEFAULT = "ap50"


def _finite(value, default=float("-inf")):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _empty_best_stat():
    return {
        "epoch": -1,
        "ap": float("-inf"),
        "ap50": float("-inf"),
        "ar": float("-inf"),
        "fitness": float("-inf"),
    }


def normalize_val_fitness_metric(raw, *, default=_VAL_FITNESS_DEFAULT) -> str:
    key = str(raw or "").strip().lower()
    if not key:
        return default
    if key in _VAL_FITNESS_ALIASES:
        return _VAL_FITNESS_ALIASES[key]
    print(
        f"val_fitness_metric={raw!r} 无效，回退 {default}；"
        f"可选: {', '.join(sorted(_VAL_FITNESS_ALIASES))}"
    )
    return default


def fitness_from_metrics(metrics, metric: str) -> float:
    """按配置从 AP/AP50/AR 合成选模标量。"""
    m = normalize_val_fitness_metric(metric)
    ap = _finite(metrics.get("ap") if metrics else None)
    ap50 = _finite(metrics.get("ap50") if metrics else None)
    ar = _finite(metrics.get("ar") if metrics else None)
    if m == "ap":
        return ap
    if m == "ar":
        return ar
    if m == "yolo":
        if ap == float("-inf") or ap50 == float("-inf"):
            return float("-inf")
        return 0.1 * ap50 + 0.9 * ap
    return ap50


def bbox_select_metrics(stats, metric: str = _VAL_FITNESS_DEFAULT):
    """从 coco_eval_bbox.stats 取出 AP / AP50 / AR，并按配置算 fitness。"""
    seq = list(stats or [])
    ap = _finite(seq[_COCO_AP] if len(seq) > _COCO_AP else None)
    ap50 = _finite(seq[_COCO_AP50] if len(seq) > _COCO_AP50 else None)
    ar = _finite(seq[_COCO_AR100] if len(seq) > _COCO_AR100 else None)
    out = {"ap": ap, "ap50": ap50, "ar": ar, "metric": normalize_val_fitness_metric(metric)}
    out["fitness"] = fitness_from_metrics(out, out["metric"])
    return out


def is_better_bbox(current, best) -> bool:
    """所选 fitness 严格更高才换（对齐 YOLO fitness > best_fitness）。"""
    return _finite(current.get("fitness") if current else None) > _finite(
        best.get("fitness") if best else None
    )


def _metric_cell(value, width=6, digits=3) -> str:
    v = _finite(value, default=None)
    if v is None:
        return f"{'-':>{width}}"
    return f"{v:{width}.{digits}f}"


def _print_val_row(metrics, *, n_images: int, improved: bool) -> None:
    n_txt = f"{n_images:6d}" if n_images >= 0 else f"{'-':>6}"
    star = " *" if improved else ""
    print(
        f"{'all':>8} {n_txt}  AP {_metric_cell(metrics.get('ap'))}  "
        f"AP50 {_metric_cell(metrics.get('ap50'))}  AR {_metric_cell(metrics.get('ar'))}  "
        f"fitness {_metric_cell(metrics.get('fitness'), 7, 4)}{star}",
        flush=True,
    )


def _print_best_row(best, *, metric_name: str) -> None:
    best_epoch = int(best.get("epoch", -1)) if best else -1
    if best_epoch < 0:
        return
    print(
        f"{'best':>8} epoch {best_epoch + 1:<4d} {metric_name} "
        f"{_metric_cell(best.get('fitness'), 7, 4)}",
        flush=True,
    )


def _jsonable_best(stat):
    out = {}
    for k, v in dict(stat or {}).items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        if not isinstance(getattr(self, "best_stat", None), dict):
            self.best_stat = _empty_best_stat()
        else:
            empty = _empty_best_stat()
            for k, v in empty.items():
                self.best_stat.setdefault(k, v)
        self.val_fitness_metric = normalize_val_fitness_metric(
            getattr(args, "val_fitness_metric", None)
        )
        self.best_stat["metric"] = self.val_fitness_metric
        self.best_stat["fitness"] = fitness_from_metrics(self.best_stat, self.val_fitness_metric)
        try:
            patience = int(getattr(args, "patience", 0) or 0)
        except (TypeError, ValueError):
            patience = 0
        patience = max(0, patience)
        print(f"val_fitness_metric: {self.val_fitness_metric} patience: {patience}")

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                epochs=args.epoches,
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device,
                epoch=epoch,
                epochs=args.epoches,
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

            metrics = bbox_select_metrics(
                test_stats.get("coco_eval_bbox"), self.val_fitness_metric
            )
            improved = is_better_bbox(metrics, self.best_stat)
            disk_best = bool(self.output_dir) and (self.output_dir / "best.pth").is_file()
            tracked = int(self.best_stat.get("epoch", -1)) >= 0
            can_write_best = tracked or not disk_best
            if improved and (not can_write_best):
                print(
                    "skip overwrite best.pth: resume checkpoint has no best_stat "
                    f"(metric={self.val_fitness_metric} fitness={metrics['fitness']:.6f})"
                )
            if improved and can_write_best:
                self.best_stat = {"epoch": epoch, **metrics}
                if self.output_dir:
                    best_path = self.output_dir / "best.pth"
                    prev_path = self.output_dir / "best_prev.pth"
                    if dist_utils.is_main_process() and best_path.is_file():
                        os.replace(str(best_path), str(prev_path))
                    dist_utils.save_on_master(self.state_dict(), best_path)
                    if dist_utils.is_main_process():
                        with (self.output_dir / "best_stat.json").open("w", encoding="utf-8") as f:
                            json.dump(_jsonable_best(self.best_stat), f, ensure_ascii=False, indent=2)

            try:
                n_images = len(self.val_dataloader.dataset)
            except Exception:
                n_images = -1
            _print_val_row(metrics, n_images=n_images, improved=improved)
            _print_best_row(self.best_stat, metric_name=self.val_fitness_metric)

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'val_select': _jsonable_best({**metrics, 'improved': improved}),
                'best_stat': _jsonable_best(self.best_stat),
                'patience': patience,
                'stale': (epoch - int(self.best_stat.get("epoch", -1)))
                if int(self.best_stat.get("epoch", -1)) >= 0
                else None,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            best_epoch = int(self.best_stat.get("epoch", -1))
            if patience > 0 and best_epoch >= 0 and (epoch - best_epoch) >= patience:
                print(
                    f"EarlyStopping: {patience} epochs without improve "
                    f"(best epoch={best_epoch} fitness={self.best_stat.get('fitness')})"
                )
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )
        metrics = bbox_select_metrics(
            test_stats.get("coco_eval_bbox"),
            getattr(self, "val_fitness_metric", None),
        )
        try:
            n_images = len(self.val_dataloader.dataset)
        except Exception:
            n_images = -1
        _print_val_row(metrics, n_images=n_images, improved=False)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
