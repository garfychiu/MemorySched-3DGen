"""批量跑感知评测：遍历 outputs/ablation/ 下所有含 run_metrics.jsonl 的物体目录，
依次调用 eval_perceptual.main() 计算 rel_lpips（幂等覆盖已有值）。

用法:
    python run_eval_all.py [--ablation_root outputs/ablation] [--only bird bulldog]
"""
import os
import sys
import argparse

import eval_perceptual as E


def main():
    ap = argparse.ArgumentParser(description='Batch perceptual eval over all ablation objects.')
    ap.add_argument('--ablation_root', default='outputs/ablation')
    ap.add_argument('--only', nargs='*', default=None,
                    help='只跑指定的物体目录名（默认全部）。')
    args = ap.parse_args()

    root = args.ablation_root
    if not os.path.isdir(root):
        raise SystemExit(f'No ablation root: {root}')

    subdirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, 'run_metrics.jsonl'))
    )
    if args.only:
        keep = set(args.only)
        subdirs = [d for d in subdirs if d in keep]

    if not subdirs:
        print(f'[run_eval_all] 没有找到含 run_metrics.jsonl 的物体目录于 {root}')
        return

    print(f'[run_eval_all] 将评测 {len(subdirs)} 个物体: {subdirs}')

    # 保存原始 sys.argv，逐物体覆盖后调用 eval_perceptual.main()
    orig_argv = sys.argv
    failures = []
    for d in subdirs:
        abl_dir = os.path.join(root, d)
        print('\n' + '=' * 60)
        print(f'>>> {d}')
        print('=' * 60)
        try:
            sys.argv = ['eval_perceptual.py', '--ablation_dir', abl_dir]
            E.main()
        except Exception as e:  # 单个物体失败不中断其余
            import traceback
            failures.append(d)
            print(f'[run_eval_all][ERROR] {d} 失败: {e}')
            traceback.print_exc()
    sys.argv = orig_argv

    print('\n' + '=' * 60)
    print('[run_eval_all] 完成。')
    if failures:
        print(f'  失败 {len(failures)} 个: {failures}')
    else:
        print('  全部成功。')


if __name__ == '__main__':
    main()
