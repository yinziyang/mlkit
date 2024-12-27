/*
Package pca 实现了主成分分析（Principal Component Analysis），这是一个对 sklearn.decomposition.PCA 的 Go 语言实现。
参考文档：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

主成分分析是一种降维方法，通过正交变换将可能相关的变量转换为线性不相关的变量，这些不相关的变量称为主成分。
该实现使用SVD（奇异值分解）来计算主成分。

主要特点：
  - 支持指定保留的主成分数量
  - 提供每个主成分的解释方差比
  - 支持数据的逆变换（还原）
  - 完全兼容 sklearn 的 API 设计

Python 与 Go 实现对比：

Python 版本：
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    X_restored = pca.inverse_transform(X_reduced)

Go 版本：
    pca := pca.NewPCA(2)
    reduced, _ := pca.FitTransform(X)
    restored, _ := pca.InverseTransform(reduced)

属性说明：
- Components_: 主成分（特征向量）
- Mean_: 训练数据的特征均值
- SingularValues_: SVD分解得到的奇异值
- ExplainedVarianceRatio_: 每个主成分解释的方差比例
- TotalExplainedVarianceRatio_: 保留的主成分解释的总方差比例
*/
package pca
