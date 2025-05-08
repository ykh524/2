import math, statistics, numpy as np, pandas as pd

# 为了可重复随机实验，设置随机种子
np.random.seed(42)

# 隐私参数
EPS_TOTAL = 1.0          # 总隐私预算 ε
DELTA     = 0.05         # 理论界限失败概率 δ
R         = 1000         # Monte-Carlo 重复次数
CSV_FILE  = "adult.csv"  # 数据文件路径

# 对数值属性发布差分隐私直方图
# series: Pandas Series；bins: 区间边界；eps: 隐私预算 ε
# global sensitivity = 1 (插入/删除邻居模型)
# Laplace 噪声尺度 b = 1/ε
def dp_histogram(series, bins, eps):
    counts, edges = np.histogram(series.to_numpy(), bins=bins)
    noisy = counts + np.random.laplace(scale=1/eps, size=len(counts))
    return noisy, edges

# 对类别属性发布差分隐私计数
# series: Pandas Series；eps: 隐私预算 ε
# global sensitivity = 1
# Laplace b = 1/ε
def categorical_dp_hist(series, eps):
    labels  = series.unique()
    counts  = series.value_counts().reindex(labels, fill_value=0).values
    noisy   = counts + np.random.laplace(scale=1/eps, size=len(counts))
    return noisy, labels

# 读取数据
df = pd.read_csv(CSV_FILE)

# 准备真实直方图/计数
bins_age    = np.linspace(17, 91, 11)  # 年龄 10 等宽区间
true_age, _ = np.histogram(df["age"], bins=bins_age)
lab_wc      = df["workclass"].unique()
true_wc     = df["workclass"].value_counts().reindex(lab_wc, fill_value=0).values
lab_ed      = df["education"].unique()
true_ed     = df["education"].value_counts().reindex(lab_ed, fill_value=0).values

# Scenario (i): 每个属性 ε = 1
eps_i = EPS_TOTAL
age_noisy_i, _   = dp_histogram(df["age"], bins_age, eps_i)
wc_noisy_i, lab_w = categorical_dp_hist(df["workclass"], eps_i)
ed_noisy_i, lab_e = categorical_dp_hist(df["education"], eps_i)

print("=== Scenario (i) — Laplace scale b = 1 (ε = 1) ===")
print("Noisy age histogram:    ", age_noisy_i.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_i.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_i.astype(int))))
print()

# Scenario (ii): 属性不相关时仍可并行消耗 ε，无需拆分预算
# 直接使用 ε = 1，与 Scenario (i) 保持一致
eps_each = EPS_TOTAL
age_noisy_ii, _ = dp_histogram(df["age"], bins_age, eps_each)
wc_noisy_ii, _  = categorical_dp_hist(df["workclass"], eps_each)
ed_noisy_ii, _  = categorical_dp_hist(df["education"], eps_each)

print("=== Scenario (ii) — uncorrelated，Laplace scale b = 1 (ε = 1) ===")
print("Noisy age histogram:    ", age_noisy_ii.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_ii.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_ii.astype(int))))
print()

# 验证理论上界 vs 经验统计
try:
    from tqdm import trange
except ImportError:
    trange = range

k_age, k_wc, k_ed = len(bins_age)-1, len(lab_wc), len(lab_ed)
# 理论 sup‐error 上界 (Laplace b=1): (1/ε)·ln(k/δ) = ln(k/δ)
bound_age = math.log(k_age/DELTA)
bound_wc  = math.log(k_wc/DELTA)
bound_ed  = math.log(k_ed/DELTA)

exceed_age = exceed_wc = exceed_ed = 0
mae_age = []; mae_wc = []; mae_ed = []

for _ in trange(R, desc="Monte-Carlo"):
    age_n, _ = dp_histogram(df["age"], bins_age, eps_i)
    wc_n, _  = categorical_dp_hist(df["workclass"], eps_i)
    ed_n, _  = categorical_dp_hist(df["education"], eps_i)

    err_age = np.abs(age_n - true_age)
    err_wc  = np.abs(wc_n  - true_wc)
    err_ed  = np.abs(ed_n  - true_ed)

    if err_age.max() > bound_age: exceed_age += 1
    if err_wc.max()  > bound_wc:  exceed_wc  += 1
    if err_ed.max()  > bound_ed:  exceed_ed  += 1

    mae_age.append(err_age.mean())
    mae_wc.append(err_wc.mean())
    mae_ed.append(err_ed.mean())

print(f"95% (δ={DELTA})")
print(f"age ≤ {bound_age:.2f}, workclass ≤ {bound_wc:.2f}, education ≤ {bound_ed:.2f}")

print(f"\nR")
print("ove upper bound：")
print(f"age       : {exceed_age/R:.3%}")
print(f"workclass : {exceed_wc/R:.3%}")
print(f"education : {exceed_ed/R:.3%}")

def fmt(x): return f"{statistics.mean(x):.2f} ± {statistics.stdev(x):.2f}"
print("error (± SD)：")
print(f"age       : {fmt(mae_age)}")
print(f"workclass : {fmt(mae_wc)}")
print(f"education : {fmt(mae_ed)}")
#When we treat the three releases as one composite mechanism, basic sequential composition says their total privacy loss is the sum of the individual ε’s. 
#To keep the overall budget at 1, we must spend only 1⁄3 on each query, which multiplies the Laplace scale by 3.
