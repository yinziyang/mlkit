# StandardScaler

StandardScaler是[scikit-learn StandardScaler](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html)的Go语言实现。它通过移除均值并缩放到单位方差来标准化特征。

## 标准化过程

标准化的计算公式如下：

```
z = (x - μ) / σ
```

其中：
- μ 是训练样本的均值
- σ 是训练样本的标准差

## 特性

- 支持可选的均值中心化（with_mean）
- 支持可选的方差归一化（with_std）
- 完全兼容sklearn的API设计
- 支持Fit、Transform、FitTransform和InverseTransform操作

## 使用示例

### Python版本（sklearn）

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
```

### Go版本（mlkit）

```go
import (
    "github.com/yinziyang/mlkit/preprocessing/standard_scaler"
    "gonum.org/v1/gonum/mat"
)

// 创建一个新的StandardScaler实例
scaler := standard_scaler.NewStandardScaler(true, true)

// 拟合数据
err := scaler.Fit(X)
if err != nil {
    // 处理错误
}

// 转换数据
scaled, err := scaler.Transform(X)
if err != nil {
    // 处理错误
}

// 或者使用FitTransform一步完成
scaled, err = scaler.FitTransform(X)
if err != nil {
    // 处理错误
}

// 如果需要，可以转换回原始空间
original, err := scaler.InverseTransform(scaled)
if err != nil {
    // 处理错误
}
```

## 实现细节

1. **均值计算**：
   - 当`with_mean=true`时，计算每个特征的均值
   - 存储在`Mean_`字段中

2. **标准差计算**：
   - 当`with_std=true`时，计算每个特征的标准差
   - 使用有偏估计（ddof=0）以与sklearn保持一致
   - 存储在`Scale_`字段中
   - 当特征的标准差为0时，将`Scale_`设置为1以避免除零错误

3. **数据结构**：
   - 使用`gonum/mat`包的`Dense`矩阵进行数据处理
   - 支持2D数据矩阵的处理

## 模型保存和加载

StandardScaler支持将训练好的模型保存到文件中，以及从文件中加载模型：

```go
// 保存模型
err := scaler.Save("scaler.json")
if err != nil {
    // 处理错误
}

// 加载模型
loadedScaler, err := standard_scaler.Load("scaler.json")
if err != nil {
    // 处理错误
}

// 使用加载的模型进行转换
scaled, err := loadedScaler.Transform(X)
```

模型文件使用JSON格式存储，包含以下信息：
- 特征均值（mean）
- 特征标准差（scale）
- 均值中心化设置（with_mean）
- 方差归一化设置（with_std）

## 注意事项

1. 实现采用有偏估计（ddof=0）计算标准差，与sklearn保持一致
2. 当特征的标准差为0时，会将scale_设置为1，避免除零错误
3. 在使用Transform之前必须先调用Fit方法
4. 所有输入数据必须是`*mat.Dense`类型

## 依赖

- [gonum/mat](https://pkg.go.dev/gonum.org/v1/gonum/mat) - 用于矩阵运算
