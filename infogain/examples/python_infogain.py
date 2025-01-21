from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 数据
documents = [
    "python java 编程 代码",
    "代码 开发 python 程序 测试",
    "编程 开发 测试",
    "数据 分析 python 统计",
    "机器学习 数据 分析 模型",
    "网络 服务器 安全",
    "服务器 网络 运维 监控"
]
labels = [0, 0, 0, 1, 1, 2, 2]

# Step 1: 生成词袋模型
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")  # 按单词分割
X = vectorizer.fit_transform(documents)  # 文档-词矩阵
vocab = vectorizer.get_feature_names_out()  # 提取词汇表

# Step 2: 计算数据集的熵 H(D)
def compute_entropy(labels):
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

H_D = compute_entropy(labels)

# Step 3: 计算每个词的条件熵 H(D|W)
def compute_conditional_entropy(feature, labels):
    indices_with = feature == 1
    indices_without = feature == 0
    subset_with = labels[indices_with]
    subset_without = labels[indices_without]

    # 条件熵公式
    H_D_given_W = 0
    for subset in [subset_with, subset_without]:
        if len(subset) > 0:
            H_D_given_W += (len(subset) / len(labels)) * compute_entropy(subset)
    return H_D_given_W

# 计算每个词的信息增益
info_gain = {}
for i, word in enumerate(vocab):
    feature = X[:, i].toarray().flatten()
    H_D_given_W = compute_conditional_entropy(feature, np.array(labels))
    info_gain[word] = H_D - H_D_given_W

# Step 4: 输出信息增益
for word, ig in sorted(info_gain.items(), key=lambda x: x[1], reverse=True):
    print(f"词: {word}, 信息增益: {ig:.4f}")

# 对比Go版本的结果
print("\n对比Go版本的结果:")
print("""
词: 分析, 信息增益: 0.8631
词: 数据, 信息增益: 0.8631
词: 服务器, 信息增益: 0.8631
词: 网络, 信息增益: 0.8631
词: 代码, 信息增益: 0.4696
词: 开发, 信息增益: 0.4696
词: 测试, 信息增益: 0.4696
词: 编程, 信息增益: 0.4696
词: 机器学习, 信息增益: 0.3060
词: 模型, 信息增益: 0.3060
词: 统计, 信息增益: 0.3060
词: python, 信息增益: 0.3060
词: 安全, 信息增益: 0.3060
词: 监控, 信息增益: 0.3060
词: 运维, 信息增益: 0.3060
词: java, 信息增益: 0.1981
词: 程序, 信息增益: 0.1981
""") 