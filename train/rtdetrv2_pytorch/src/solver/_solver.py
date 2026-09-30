"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None 
    return m.to(device) 


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 

    def _setup(self, ):
        """Avoid instantiating unnecessary classes 
        """
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model
        
        # NOTE (lyuwenyu): must load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(self.model.to(device), sync_bn=cfg.sync_bn, \
            find_unused_parameters=cfg.find_unused_parameters)

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch
        
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

    def cleanup(self, ):
        if self.writer:
            atexit.register(self.writer.close)

    def train(self, ):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self, ):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        self.evaluator = self.cfg.evaluator
        
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                v.to(device)

    def state_dict(self):
        """state dict, train/eval
        """
        state = {}
        state['date'] = datetime.now().isoformat()
        
        # TODO for resume
        state['last_epoch'] = self.last_epoch
        if getattr(self, 'best_stat', None) is not None:
            state['best_stat'] = dict(self.best_stat)

        for k, v in self.__dict__.items():
            if k == 'best_stat':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict() 

        return state


    def load_state_dict(self, state):
        """load state dict, train/eval
        """
        # TODO
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        loaded_best = state.get('best_stat')
        if isinstance(loaded_best, dict):
            self.best_stat = dict(loaded_best)
            print('Load best_stat')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')


    def load_resume_state(self, path: str):
        """load resume
        """
        # for cuda:0 memory
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        self.load_state_dict(state)

    
    def load_tuning_state(self, path: str,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist_utils.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        unmatched = list(infos.get("unmatched") or [])
        missed = list(infos.get("missed") or [])
        print(
            "Load model.state_dict, matched={} missed={} unmatched={}".format(
                len(stat), len(missed), len(unmatched)
            )
        )
        skipped = [k for k in unmatched if self._is_expected_tuning_unmatched(k)]
        if skipped:
            print(f"unmatched skipped (class head / anchors): {skipped}")
        self._assert_tuning_architecture(infos)

    @staticmethod
    def _is_expected_tuning_unmatched(key: str) -> bool:
        """COCO 预训练微调：分类头类别数变化；imgsz 变化时 anchors/valid_mask 形状不同。骨干/encoder 必须对齐。"""
        if key in {"decoder.anchors", "decoder.valid_mask"}:
            return True
        return any(token in key for token in ("score_head", "denoising_class_embed", "class_embed"))

    @classmethod
    def _assert_tuning_architecture(cls, infos: Dict[str, list]) -> None:
        unmatched = list(infos.get("unmatched") or [])
        missed = list(infos.get("missed") or [])
        struct_unmatched = [k for k in unmatched if not cls._is_expected_tuning_unmatched(k)]
        struct_missed = [
            k
            for k in missed
            if not k.endswith("num_batches_tracked")
            and (k.startswith("backbone.") or k.startswith("encoder."))
        ]
        if not struct_unmatched and not struct_missed:
            return
        sample = ", ".join((struct_unmatched + struct_missed)[:8])
        raise RuntimeError(
            "tuning 与模型结构不匹配（unmatched_struct={} missed_struct={}）。"
            "请确认 model_yml 与预训练同骨干：rtdetrv2_r50vd_6x_coco_ema.pth 须用 "
            "configs/rtdetrv2/rtdetrv2_r50vd_scanly.yml（不要用 r34/r18 或 r50vd_m）。"
            " 样例: {}".format(len(struct_unmatched), len(struct_missed), sample)
        )

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')


    def val(self, ):
        raise NotImplementedError('')
