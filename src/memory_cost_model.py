"""
memory_cost_model.py — 统一显存代价模型与自适应调度（支柱二 + 支柱三）。

把 profile_lod.py 拟合出的动态 LOD 立方代价模型
    M_extract(G) = β_ext + α · G^3
与两个显存杠杆（FP8 量化 / Triplane 栅格分块）封装为一个可在 run.py --auto
中直接调用的调度器：给定设备显存预算 B，自动选定
    * FP8 量化开关     —— UNet 权重减半（Stage-1 主杠杆），代价是 Stage-2 ~0.82GB
                          反量化尖峰，净收益为正；
    * 栅格分块粒度     —— --Chunk，把 Stage-2 峰值从 G^3 解耦为常量 g，
                          从而让高 LOD（64/128）能在低显存卡上跑通；
    * 网格分辨率 G*    —— 闭环查表：分块路径下 B<7.5GB→G64、7.5<=B<15.5GB→G128
                          （含 0.5GB 探测容差，使 7.96GB 的 8GB 卡归入 G128 档）；
                          无分块路径下统一 G128（显存已足够容纳 G^3 峰值）。
全程不重训（training-free）。

系数默认嵌入论文量级的经验值；若有 profile_lod.py 输出的拟合 JSON，可用
load_cost_model() 覆盖以精确匹配当前硬件。

用法：
    python -m src.memory_cost_model --budget_gb 8      # 自检某预算下的工作点
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass


@dataclass
class CostModel:
    alpha: float       # 每单位 G^3 顶点的显存系数 (GB / G^3)
    beta_ext: float    # 与 G 无关的固定开销 (GB)：TriplaneTransformer + planes 常驻
    beta_fp: float     # forward_planes 峰值（Stage-2 常量开销下界，GB）
    delta_fp8: float   # FP8 反量化缓冲尖峰 (GB)，近似常量
    r2: float = 1.0
    source: str = "embedded-default"


# 论文量级的默认系数（与 profile_lod.py 叙事一致：G=96 提取峰值≈9.12GB，
# G=128≈15GB，FP8 反量化尖峰≈0.82GB）。可由 profile_lod 输出的 JSON 覆盖。
# 校验：β_ext + α·96^3 = 4.83 + 4.85e-6·884736 ≈ 9.12GB；
#       β_ext + α·128^3 = 4.83 + 4.85e-6·2097152 ≈ 15.0GB。
DEFAULT_COST_MODEL = CostModel(
    alpha=4.85e-6,
    beta_ext=4.83,
    beta_fp=4.0,
    delta_fp8=0.82,
)

FP8_BUDGET_THRESHOLD_GB = 12.0   # ≤12GB 视为消费级小卡，开启 FP8
# --Chunk 级别对应的真实分块顶点数（Stage-2 FlexiCubes MLP 分块栅格提取的块大小，单位：顶点数；
# 见 CHUNK_LEVEL_TO_VERTS）。与 optimization_plan.md / 实测一致：级别 2（=200000）为性价比最优档
# （LOD128 总峰值 6.12GB）。注意：6GB 等极紧显存档的瓶颈在 Stage-1（扩散），Stage-2 分块对总峰值
# 几乎无影响，故各分块档统一取 200000。级别 0 = 关闭分块（整批，仅 16/24GB 充裕显存档用）。
GRID_EXTRACT_CHUNK_DEFAULT = 200000  # 所有分块档统一取值

# ── --Chunk 级别 → Stage-2 栅格分块顶点数（替代原 --grid_extract_chunk 原始顶点数）──
# 级别 0=关闭分块（整批，峰值随 G^3，仅 16/24GB 充裕显存档）；1/2/3 为分块档，
# 数值越小分块越细、Stage-2 峰值越低但越慢。级别 2（=200000）为论文实测性价比最优档
# （LOD128 总峰值 6.12GB）；1/3 为推断档，未逐一实测。
CHUNK_LEVEL_TO_VERTS = {
    0: 0,        # 关闭分块（整批，速度优先，仅 16/24GB 充裕显存档）
    1: 100000,   # 激进分块：最低 Stage-2 峰值，最慢
    2: GRID_EXTRACT_CHUNK_DEFAULT,  # 默认分块档（性价比最优，LOD128 总峰值 6.12GB）
    3: 400000,   # 轻度分块：峰值略高，速度优先
}

def chunk_level_to_verts(level: int) -> int:
    """把 --Chunk 级别映射为实际分块顶点数（等价原 --grid_extract_chunk 数值）。"""
    return CHUNK_LEVEL_TO_VERTS.get(int(level), GRID_EXTRACT_CHUNK_DEFAULT)


# ── 均衡点分块 (--Lambda，动态由 Stage1 反解) ─────────────────────────────
# 核心：Chunk 不是写死常量，而是根据该 LOD 下 Stage1 的实际显存动态反解，
# 使 Stage2 峰值 ≈ Stage1（全局峰值锁死在 Stage1 刚性下限，不 OOM 也不浪费）。
#
# 分块域（λ<1）Stage2 峰值模型（六点实测标定 + 物理公式校正）：
#   M_stage2(g, G) = β_eff(G) + α_vert · g      (g=分块顶点数, G=grid_res/LOD)
# β_eff(G) = 分块域固定基线（recon Transformer 权重 + FlexiCubes 体素网格 +
#           CUDA 上下文 + 逐顶点采样特征 M_sample(G)），随 G³ 增长；
# α_vert   = 每块顶点 weight MLP 解码增量系数，物理上 = F·b = 1920·4 bytes
#           ≈ 7.15e-6 GB/顶点，与 G 无关（各 LOD 共用同一值）。
#
# 标定点（FP8 + 单输入，Stage-2 峰值 torch.cuda.max_memory_allocated）：
#   LOD128: 6 点最小二乘 (g=347636→7.25, 800k→10.62, 833k→10.88,
#           848200→11.00, 1101k→12.86, 1482k→15.76)
#           → β_eff=4.64, α=7.50e-6（残差≤0.04 GB，与理论 F·b=7.15e-6 一致）
#   LOD96 : 2 点 (g=200000→4.77, 258000→5.20) → β_eff=3.29, α=7.41e-6
#   LOD64 : 1 点 (g=200000→4.07) + 共用 α → β_eff=2.64
# 取代旧模型 β_ext=1.44 + α_eff(G)·g（旧 β_ext 错借整批域截距 1.44，导致大 g 预测虚高近 3 倍；
# 详见 optimization_plan.md 附录 A 3.3.6）。
#
# Stage1 显存本身随 LOD 变化（低 LOD 略降），由用户实测两点标定：
#   LOD128: Stage1=5.86    LOD96: Stage1=5.72
#   → M_stage1(G) = 5.618 + 1.154e-7 · G³   （线性拟合，弱 G³ 相关）
#
# 动态均衡点分块顶点数（绝不写死）：
#   g_eq(G) = (M_stage1(G) − β_eff(G)) / α_vert
# 临界均衡因子 λ0 的数学由来：
#   λ0(G) = g_eq(G) / G³ = (M_stage1(G) − β_eff(G)) / (α_vert · G³)
# 即 λ0 由“该 LOD 的动态 Stage1”联立解出；再缩块不降全局峰值仅增耗时。
BETA_EFF_BY_LOD = {64: 2.64, 96: 3.29, 128: 4.64}   # 分块域固定基线 β_eff(G)，随 LOD 增长
# 分块每顶点增量系数 α_vert 标定点（LOD=G → α）。与 G 基本无关（物理上 = F·b ≈ 7.15e-6），
# 逐 LOD 微调以贴合实测。每个 LOD 真跑过一次后，run.py 会用实测 Stage2 反推 α，
# 由 persist_alpha() 直接写回对应 LOD 的值——之后 import 即真实值，无需手改。
ALPHA_EFF_BY_LOD = {64: 7.1500e-06, 96: 7.4100e-06, 128: 7.3935e-06}


def persist_alpha(grid_res: int, alpha: float) -> bool:
    """把实测 α_eff 直接写回本文件的 ALPHA_EFF_BY_LOD 字典（键=LOD）。

    先更新内存中的字典，再用正则把源码里 `ALPHA_EFF_BY_LOD = {...}` 那一行重写为
    最新值（按 LOD 升序、科学计数）。下次运行 import 时即为真实值，彻底告别
    “写死→手改”与独立缓存文件。写文件失败不影响本次运行，返回 False。
    """
    try:
        ALPHA_EFF_BY_LOD[int(grid_res)] = float(alpha)  # 先更新内存
    except Exception:
        return False
    try:
        src_path = os.path.abspath(__file__)
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
        items = ", ".join(f"{k}: {v:.4e}" for k, v in sorted(ALPHA_EFF_BY_LOD.items()))
        new_line = f"ALPHA_EFF_BY_LOD = {{{items}}}"
        new_text, n = re.subn(r"ALPHA_EFF_BY_LOD = \{[^}]*\}", new_line, text, count=1)
        if n == 1 and new_text != text:
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            return True
    except Exception:
        pass
    return False
# Stage1 显存随 LOD 标定：M_stage1(G) = STAGE1_A + STAGE1_B·G³（5.86@128, 5.72@96）
STAGE1_A = 5.618
STAGE1_B = 1.154e-7
STAGE1_PEAK_EQ_GB = 5.86      # 参考：LOD128 基准 Stage1（论文引用）


def predict_stage1(grid_res: int) -> float:
    """该 LOD 下 Stage1 峰值显存预测（GB）：M = STAGE1_A + STAGE1_B·G³。"""
    return STAGE1_A + STAGE1_B * (grid_res ** 3)


def alpha_eff(grid_res: int) -> float:
    """分块每顶点增量系数 α_eff(G)。

    直接读 ALPHA_EFF_BY_LOD：命中该 LOD 即返回其（已被实测反推写回的）真实值；
    未命中则按 G 在已有标定点间线性插值/外推。每个 LOD 真跑一次后，persist_alpha()
    会把实测值写回字典，故这里读到的即真值——无需缓存文件。
    """
    lods = sorted(ALPHA_EFF_BY_LOD)
    if grid_res in ALPHA_EFF_BY_LOD:
        return ALPHA_EFF_BY_LOD[grid_res]
    if grid_res < lods[0]:
        a_lo, a_hi = ALPHA_EFF_BY_LOD[lods[0]], ALPHA_EFF_BY_LOD[lods[1]]
        slope = (a_hi - a_lo) / (lods[1] - lods[0])
        return max(a_lo + slope * (grid_res - lods[0]), 1e-7)
    a_lo, a_hi = ALPHA_EFF_BY_LOD[lods[-2]], ALPHA_EFF_BY_LOD[lods[-1]]
    slope = (a_hi - a_lo) / (lods[-1] - lods[-2])
    return max(a_hi + slope * (grid_res - lods[-1]), 1e-7)


def beta_eff(grid_res: int) -> float:
    """分块域固定基线 β_eff(G)，随 LOD 增长（含 M_sample(G) 等固定+采样开销）。

    直接读 BETA_EFF_BY_LOD：命中该 LOD 即返回其实测基线；未命中则按 G 在已有
    标定点间线性插值/外推。取代旧模型里与 G 无关的固定 β_ext=1.44（该值实为整批域
    截距，错用于分块域导致大 g 预测虚高近 3 倍）。
    """
    lods = sorted(BETA_EFF_BY_LOD)
    if grid_res in BETA_EFF_BY_LOD:
        return BETA_EFF_BY_LOD[grid_res]
    if grid_res < lods[0]:
        b_lo, b_hi = BETA_EFF_BY_LOD[lods[0]], BETA_EFF_BY_LOD[lods[1]]
        slope = (b_hi - b_lo) / (lods[1] - lods[0])
        return max(b_lo + slope * (grid_res - lods[0]), 0.0)
    b_lo, b_hi = BETA_EFF_BY_LOD[lods[-2]], BETA_EFF_BY_LOD[lods[-1]]
    slope = (b_hi - b_lo) / (lods[-1] - lods[-2])
    return max(b_hi + slope * (grid_res - lods[-1]), 0.0)


def predict_stage2_extract(g: int, grid_res: int) -> float:
    """分块域 Stage2 峰值显存预测（GB）：M = β_eff(G) + α_eff(G)·g。"""
    return beta_eff(grid_res) + alpha_eff(grid_res) * g


def auto_chunk_verts(grid_res: int, m_stage1: float | None = None) -> int:
    """动态均衡点分块顶点数：由该 LOD 的 Stage1 反解 g_eq = (Stage1 − β_eff(G))/α_eff(G)。

    使 Stage2 峰值 ≈ Stage1 刚性下限（M_stage1），全局峰值锁在 Stage1 之下、
    绝不 OOM 也不浪费显存；再缩块不降峰值仅增推理耗时。g 不超过 G³
    （分块不能大于总顶点数，达上限即退化为整批）。绝不写死为固定值。

    求解直接取动态均衡点（无额外安全余量），并将取整改为向下取整到千，
    确保“实测” Stage2 严格 ≤ Stage1（历史 bug：用偏低的口头 α 锚点 + 向上取整，
    导致算出 chunk 偏大、Stage2 冲过 Stage1）。
    """
    m1 = m_stage1 if m_stage1 is not None else predict_stage1(grid_res)
    a = alpha_eff(grid_res)
    g = (m1 - beta_eff(grid_res)) / a
    g = min(max(g, 0.0), float(grid_res ** 3))  # clamp 到 [0, G³]
    # 向下取整到千，保证 Stage2 不会因取整上偏而越过 Stage1（数值也更整洁）。
    return int(math.floor(g / 1000.0) * 1000)


def lambda0_eq(grid_res: int, m_stage1: float | None = None) -> float:
    """临界均衡因子 λ0(G) 的数学由来：λ0 = (M_stage1(G) − β_eff(G)) / (α_eff(G)·G³)。"""
    m1 = m_stage1 if m_stage1 is not None else predict_stage1(grid_res)
    return (m1 - beta_eff(grid_res)) / (alpha_eff(grid_res) * (grid_res ** 3))

G_MIN, G_MAX = 16, 128
SAFETY_MARGIN_GB = 0.5           # 为 CUDA 碎片/非峰值分配预留
# 显存探测容差：消费级显卡上报的 total_memory 通常比标称值略小（如 8GB 卡报 7.96GB），
# 在档位边界比较时允许小幅容差，避免把 8GB 卡误判为 6GB 档、16GB 卡误判为分块档。
DEVICE_TIER_TOLERANCE_GB = 0.5


def load_cost_model(json_path: str) -> CostModel:
    """从 profile_lod.py 的输出 JSON 加载拟合系数。"""
    with open(json_path, "r", encoding="utf-8") as f:
        r = json.load(f)
    fit = r["fit"]
    return CostModel(
        alpha=fit["alpha_gb_per_G3"],
        beta_ext=fit["beta_ext_gb"],
        beta_fp=r.get("forward_planes_peak_gb", DEFAULT_COST_MODEL.beta_fp),
        delta_fp8=r["config"].get("delta_fp8_gb", DEFAULT_COST_MODEL.delta_fp8),
        r2=fit.get("r2", 1.0),
        source=json_path,
    )


def g_star_for_budget(cm: CostModel, budget_gb: float, fp8: bool = False,
                      margin_gb: float = SAFETY_MARGIN_GB,
                      g_min: int = G_MIN, g_max: int = G_MAX) -> int:
    """自适应 LOD 准则：解 β_ext + α·G^3 + δ_fp8 + margin ≤ B。

    返回满足预算的最大 G，clamp 到 [g_min, g_max]。
    """
    delta = cm.delta_fp8 if fp8 else 0.0
    cap = budget_gb - delta - cm.beta_ext - margin_gb
    if cap <= 0:
        return g_min
    g = int(math.floor((cap / cm.alpha) ** (1.0 / 3.0)))
    return max(g_min, min(g, g_max))


def recommend_config(budget_gb: float, cm: CostModel | None = None,
                     margin_gb: float = SAFETY_MARGIN_GB) -> dict:
    """支柱三·自适应调度闭环：给定显存预算 B，返回一组免训练工作点。

    决策逻辑（忠实于三支柱框架，依据实测 Recorded group
    "FP8+LOD64+SimplifyTrue -> peak 5.65GB" 校准的闭环表）：
      1) FP8 开关：消费级小卡（B<=12GB）开启——UNet 权重减半（Stage-1 主杠杆）。
      2) 栅格分块：预算紧张（B<16GB）时开启 grid_extract_chunk，把 Stage-2 峰值
         从 G^3 解耦为常量（实测峰值≈5.65GB，与 LOD 无关），从而让高 LOD 在低显存卡
         上跑通；预算充裕（B>=16GB）则关闭分块（默认路径，速度优先）。
      3) G*（网格分辨率）：分块路径下（含 0.5GB 探测容差），
            B<7.5GB   -> G64  （6GB 卡保守留安全余量，实测 5.65GB 峰值贴近上限）；
            7.5<=B<15.5GB -> G128 （含 8GB/12GB 卡；chunk 解耦，5.65GB 峰值余量充足）；
         无分块路径下统一取 G128（显存已足够容纳 G^3 峰值）。
      4) 视角数不调度——多视图裁剪属质量/鲁棒性受控消融变量，非显存主杠杆。

    返回 dict：{fp8_unet, grid_res, grid_extract_chunk, g_star_no_chunk,
                predicted_stage2_peak_gb, decision}
    """
    cm = cm or DEFAULT_COST_MODEL
    fp8 = budget_gb <= FP8_BUDGET_THRESHOLD_GB
    # 档位边界加探测容差：7.96GB(8GB 卡) -> 第二档(G128)，15.9GB(16GB 卡) -> 无分块档。
    use_chunk = budget_gb < 16.0 - DEVICE_TIER_TOLERANCE_GB

    if use_chunk:
        grid_res = 64 if budget_gb < 8.0 - DEVICE_TIER_TOLERANCE_GB else 128
        # Stage-2 分块对各档统一取 200000（6GB 档瓶颈在 Stage-1，调小 chunk 对总峰值无益）
        chunk = GRID_EXTRACT_CHUNK_DEFAULT
    else:
        grid_res = 128
        chunk = 0   # 0 = 关闭分块（整批，速度优先，仅 16/24GB 充裕显存档）

    # 预测 Stage-2 峰值（顺序发生，不与 Stage-1 叠加；见 run.py:411 释放 pipeline）
    if use_chunk:
        # 分块后峰值与 G 解耦，收敛到常量下界（实测 FP8+LOD64+SimplifyTrue=5.65GB）。
        # 该值为实测峰值，直接采用、不加安全余量。
        pred = max(cm.beta_fp, cm.beta_ext) + cm.delta_fp8 * fp8
    else:
        # 无分块走立方代价模型 β_ext+α·G^3，外推不确定性大，保留安全余量。
        pred = (max(cm.beta_fp, cm.beta_ext + cm.alpha * (grid_res ** 3))
                + cm.delta_fp8 * fp8 + margin_gb)
    pred = round(pred, 2)
    over_budget = pred > budget_gb + 1e-6

    decision = (
        f"budget={budget_gb}GB -> --FP8Unet={'on' if fp8 else 'off'}, "
        f"--Chunk={chunk}, --LOD(G*)={grid_res}; "
        f"predicted Stage2 peak≈{pred}GB"
        + ("  [WARNING: predicted peak exceeds budget!]" if over_budget else "")
    )
    return {
        "fp8_unet": fp8,
        "grid_res": grid_res,
        "grid_extract_chunk": chunk,
        "g_star_no_chunk": grid_res,
        "predicted_stage2_peak_gb": pred,
        "decision": decision,
    }


def detect_device_budget_gb(device: int = 0) -> float:
    """从 GPU 总显存自动推导预算 B（GB）。

    取当前卡的总 VRAM 作为预算上限（调度器内部已含 SAFETY_MARGIN 留量）。
    无 CUDA 时回退到 0.0，调用方据此提示/回退。
    """
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(device).total_memory  # bytes
    return round(total / (1024 ** 3), 2)


def _main():
    ap = argparse.ArgumentParser(description="Self-test of the adaptive memory scheduler.")
    ap.add_argument("--budget_gb", type=float, default=8.0,
                    help="Target GPU memory budget (GB).")
    ap.add_argument("--cost_model_json", type=str, default="",
                    help="Optional profile_lod.py fit JSON to override coefficients.")
    ap.add_argument("--margin_gb", type=float, default=SAFETY_MARGIN_GB)
    a = ap.parse_args()
    cm = DEFAULT_COST_MODEL
    if a.cost_model_json:
        cm = load_cost_model(a.cost_model_json)
    rec = recommend_config(a.budget_gb, cm=cm, margin_gb=a.margin_gb)
    print("CostModel:", f"alpha={cm.alpha:.3e}, beta_ext={cm.beta_ext:.2f}, "
          f"beta_fp={cm.beta_fp:.2f}, delta_fp8={cm.delta_fp8}, R2={cm.r2:.4f} "
          f"(source={cm.source})")
    print("[auto-scheduler]", rec["decision"])


if __name__ == "__main__":
    _main()
