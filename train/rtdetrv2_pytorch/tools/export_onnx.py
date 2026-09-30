"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig, yaml_utils
from src.misc.onnx_export import export_to_onnx


def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume:
        checkpoint = args.resume
    else:
        checkpoint = None
        print('not load model.state_dict, use default init state dict...')

    export_to_onnx(
        cfg,
        args.output_file,
        args.input_size,
        checkpoint=checkpoint,
        check=args.check,
        simplify=args.simplify,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config')

    args = parser.parse_args()

    main(args)
