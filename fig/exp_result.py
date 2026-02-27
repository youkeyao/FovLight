import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Make fonts larger for readability
FONT_SIZE = 12
ANNOTATION_FONTSIZE = 12
TICK_FONTSIZE = 10
CB_FONTSIZE = 10
plt.rcParams.update({'font.size': FONT_SIZE})

# 原始成对比较胜出次数矩阵
strategies = ["strategy_1", "strategy_4", "strategy_2", "strategy_3"]

data = [
    [np.nan, 110, 105, 104],
    [19, np.nan, 45, 47],
    [23, 85, np.nan, 61],
    [24, 82, 70, np.nan]
]

df = pd.DataFrame(data, index=strategies, columns=strategies)

prob = np.zeros_like(df.values, dtype=float)

for i in range(len(strategies)):
    for j in range(len(strategies)):
        if i == j:
            prob[i, j] = np.nan
        else:
            wij = df.iloc[i, j]
            wji = df.iloc[j, i]
            prob[i, j] = wij / (wij + wji)

prob_df = pd.DataFrame(prob, index=strategies, columns=strategies)

# 绘图
plt.figure(figsize=(6, 5))
im = plt.imshow(prob_df, cmap="coolwarm", vmin=0, vmax=1)

# 数值标注
for i in range(len(strategies)):
    for j in range(len(strategies)):
        if not np.isnan(prob_df.iloc[i, j]):
            plt.text(j, i, f"{prob_df.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=10)

plt.xticks(range(len(strategies)), strategies, rotation=45, ha="right")
plt.yticks(range(len(strategies)), strategies)

plt.xlabel("Compared strategy (column)")
plt.ylabel("Preferred strategy (row)")

cbar = plt.colorbar(im)
cbar.set_label("Preference probability")

# plt.title("Pairwise preference matrix (win probability)")
plt.tight_layout()
plt.savefig("raw_preference_matrix.png")