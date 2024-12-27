/*
Package standard_scaler 实现了特征标准化处理，这是一个对 sklearn.preprocessing.StandardScaler 的 Go 语言实现。
参考文档：https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html

标准化过程：

	z = (x - μ) / σ

其中：
- μ 是训练样本的均值
- σ 是训练样本的标准差

主要特点：
  - 支持可选的均值中心化（with_mean）
  - 支持可选的方差归一化（with_std）
  - 完全兼容 sklearn 的 API 设计

Python 与 Go 实现对比：

Python 版本：

	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler(with_mean=True, with_std=True)
	X_scaled = scaler.fit_transform(X)

Go 版本：

	scaler := standard_scaler.NewStandardScaler(true, true)
	scaler.Fit(X)
	scaled, _ := scaler.Transform(X)
*/
package standard_scaler
