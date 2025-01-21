# InfoGain 示例

本目录包含了信息增益特征选择算法的不同实现示例，用于对比和验证。

## Python版本 (python_infogain.py)

使用scikit-learn实现的信息增益计算。主要步骤：
1. 使用CountVectorizer生成词袋模型
2. 计算数据集的熵 H(D)
3. 计算每个词的条件熵 H(D|W)
4. 计算信息增益 IG = H(D) - H(D|W)

### 运行要求
```bash
pip install scikit-learn numpy
```

### 运行方式
```bash
python python_infogain.py
```

## Go版本

Go版本的实现在主包中，这个示例用于验证Go实现的正确性。两个版本的计算结果应该是一致的。

### 主要区别
1. Go版本使用了并发处理来提高性能
2. Go版本支持按类别选择特征
3. Go版本实现了模型的保存和加载功能 