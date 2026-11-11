"""
循环跑 GSO_300，只输出「Baseline」（24G 服务器、不加任何优化 flag 的全质量参考），
mesh 落到 outputs/ablation/meshes/Baseline/<oid>.obj，供 eval_absolute.py 与 Ours 对照、并各自对齐 GT。

输入约定：直接从 inputs/_gso_staging/<oid>.jpg 读取（这些文件已预先准备好——
          即每个物体 thumbnails/0.jpg 复制并重命名为 <oid>.jpg，stem == object_id）。
          run.py 用输入文件名 stem 当 object_id（run.py:499: name = basename.split('.')[0]），
          所以必须保证 stem == oid，否则 mesh 存成 0.obj，
          与 GT 目录 <gt_dir>/<oid>/meshes/model.obj 对不上 → eval_absolute 全部跳过。

与 run_ours_gso.py 的区别：
    - 组名固定 Baseline（不加 FP8/Chunk/LOD/Lambda，等价于原始 InstantMesh 默认配置）。
    - 默认 --view 6，无其它优化；24G 显存充足，Stage-2 全批量提取也不会 OOM。

用法（服务器上）：
    # 先小批量验证
    python run_baseline_gso.py --max_inputs 8

    # 跑全部（Baseline 组）
    python run_baseline_gso.py --staging_dir inputs/_gso_staging

    # 重跑 / 单独补跑
    python run_baseline_gso.py --force
    python run_baseline_gso.py --only 3D_Dollhouse_Sofa,xxx

跑完后与 Ours 一起评测（两组 mesh 同机的机器上）：
    python eval_absolute.py --gt_dir <GT_DIR> --dataset gso ^
        --pred_root outputs/ablation/meshes ^
        --out outputs/absolute/metrics.json --csv outputs/absolute/per_object.csv ^
        --tau 0.2 --tau_extra 0.1 --icp
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # 用同一个 venv 跑子进程

DEFAULT_CONFIG = 'configs/instant-mesh-large.yaml'
DEFAULT_STAGING_DIR = os.path.join('inputs', '_gso_staging')
DEFAULT_GT_DIR = r'D:\AR_3DGeneres\GSO_300'
DEFAULT_ABLATION_DIR = 'outputs/ablation'
GROUP_NAME = 'Baseline'

# Baseline：不加任何优化 flag（等价于原始 InstantMesh 默认配置；--view 6 即默认值，写明仅示意）。
BASELINE_EXTRA = ['--view', '6']

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def iter_staged(staging_dir):
    """遍历 staging 目录下的 <oid>.jpg（stem == oid == object_id）。"""
    objs = []
    if not os.path.isdir(staging_dir):
        print(f'[ERROR] staging_dir 不存在：{staging_dir}')
        return objs
    for f in sorted(os.listdir(staging_dir)):
        if f.lower().endswith(IMG_EXTS):
            oid = os.path.splitext(f)[0]
            objs.append((oid, os.path.join(staging_dir, f)))
    if not objs:
        print(f'[WARN] {staging_dir} 下没有任何图片（期望 <oid>.jpg）。')
    return objs


def main():
    p = argparse.ArgumentParser(description='循环跑 GSO_300 staging，只输出 Baseline（24G 服务器、无优化）。')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--staging_dir', default=DEFAULT_STAGING_DIR,
                   help='已准备好的输入目录，内含 <oid>.jpg（stem==oid）')
    p.add_argument('--gt_dir', default=DEFAULT_GT_DIR, help='评测用 GT 根目录（仅文档，供 eval 用）')
    p.add_argument('--ablation_dir', default=DEFAULT_ABLATION_DIR, help='mesh/指标记录父目录')
    p.add_argument('--max_inputs', type=int, default=0, help='最多跑前 N 个物体（0=全部）')
    p.add_argument('--only', default='', help='只跑这些 oid，逗号分隔（覆盖 --max_inputs）')
    p.add_argument('--force', action='store_true', help='忽略已生成的 mesh，强制重跑')
    p.add_argument('--save_video', action='store_true', help='额外保存环绕视频（慢，默认关）')
    p.add_argument('--continue_on_error', action='store_true', default=True,
                   help='某物体失败也继续后续（默认开）')
    p.add_argument('--no_continue_on_error', dest='continue_on_error', action='store_false',
                   help='某物体失败立即退出')
    args = p.parse_args()

    objs = iter_staged(args.staging_dir)
    if not objs:
        print('没有可用物体，退出。')
        return

    if args.only.strip():
        wanted = {w.strip() for w in args.only.split(',') if w.strip()}
        objs = [(oid, t) for (oid, t) in objs if oid in wanted]
        if not objs:
            print(f'[WARN] --only 指定的 oid 都不在 {args.staging_dir} 中。')
            return

    if args.max_inputs and args.max_inputs > 0 and not args.only.strip():
        objs = objs[:args.max_inputs]

    # 输出组目录（eval 实际读取的位置）
    mesh_root = os.path.join(args.ablation_dir, 'meshes', GROUP_NAME)
    os.makedirs(mesh_root, exist_ok=True)

    print(f'目标组：{GROUP_NAME}  ->  {mesh_root}')
    print(f'输入目录：{args.staging_dir}')
    print(f'共 {len(objs)} 个物体：')
    for oid, _ in objs:
        print('  -', oid)

    done = 0
    skipped = 0
    failed = []
    for oid, img_path in objs:
        out_obj = os.path.join(mesh_root, f'{oid}.obj')
        if os.path.exists(out_obj) and not args.force:
            print(f'\n[skip] {oid}：mesh 已存在，跳过（--force 强制重跑）')
            skipped += 1
            continue

        # 直接调 run.py，固定组名 Baseline（输入文件 stem == oid，mesh 即 <oid>.obj）
        cmd = [PY, 'run.py', args.config, img_path,
               '--ablation_name', GROUP_NAME,
               '--ablation_dir', args.ablation_dir] + BASELINE_EXTRA
        if args.save_video:
            cmd.append('--save_video')

        print(f'\n===== [{oid}] ({done + 1}/{len(objs)}) =====')
        print(' '.join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f'[ERROR] {oid} 失败，返回码 {rc}')
            failed.append(oid)
            if not args.continue_on_error:
                sys.exit(rc)
            continue
        done += 1

    print(f'\n===== 完成 =====')
    print(f'  成功：{done}  跳过(已存在)：{skipped}  失败：{len(failed)}')
    if failed:
        print('  失败列表：', ', '.join(failed))
        print('  用 --force 重跑失败项，或用 --only <oid1>,<oid2> 单独补跑。')


if __name__ == '__main__':
    main()
