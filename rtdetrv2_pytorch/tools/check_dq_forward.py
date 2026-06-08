import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


def check_output(out):
    print('output keys:', sorted(out.keys()))
    assert 'pred_logits' in out
    assert 'pred_boxes' in out
    print('pred_logits:', tuple(out['pred_logits'].shape))
    print('pred_boxes:', tuple(out['pred_boxes'].shape))
    if 'count_logits' in out:
        print('count_logits:', tuple(out['count_logits'].shape))
    if 'density_map' in out:
        print('density_map:', tuple(out['density_map'].shape))
    if 'dynamic_query_num' in out:
        print('dynamic_query_num:', int(out['dynamic_query_num'].item()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    model = cfg.model.to(args.device).eval()
    x = torch.randn(args.batch_size, 3, args.size, args.size, device=args.device)
    out = model(x, targets=None)
    check_output(out)


if __name__ == '__main__':
    main()
