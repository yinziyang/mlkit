# MLKit

MLKit 是一个用 Go 语言实现的机器学习工具包，旨在提供高效且易于使用的机器学习算法。它的目标是提供类似于 Python 的 scikit-learn 的功能，同时利用 Go 的性能和类型安全性。

## 功能

目前，MLKit 实现了以下算法：

### 降维

#### PCA (主成分分析)
- 使用 PCA 进行降维
- 与 scikit-learn 的 PCA 实现兼容
- 功能：
  - 可配置的组件数量
  - 解释方差比计算
  - 模型持久化（保存/加载）
  - 转换和逆转换操作

### 预处理

#### StandardScaler
- 使用标准化进行特征缩放
- 与 scikit-learn 的 StandardScaler 兼容
- 功能：
  - 均值归一化（可选）
  - 方差缩放（可选）
  - 拟合/转换接口
  - 模型持久化（保存/加载）

## 安装

```bash
go get github.com/yinziyang/mlkit
```

## 使用示例

### PCA 示例

#### Go 代码

```go
package main

import (
    "fmt"
    "github.com/yinziyang/mlkit/decomposition/pca"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // 创建样本数据
    data := mat.NewDense(3, 4, []float64{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    })

    // 初始化 PCA，设置为 2 个组件
    pca := pca.NewPCA(2)
    
    // 拟合并转换数据
    transformed := pca.FitTransform(data)
    
    // 获取解释方差比
    fmt.Printf("Explained variance ratio: %v\n", pca.ExplainedVarianceRatio())
    
    // 保存模型
    if err := pca.Save("pca_model.gob"); err != nil {
        panic(err)
    }
}
```

#### Python 代码

```python
import numpy as np
from sklearn.decomposition import PCA

# 创建样本数据
X = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# 初始化 PCA，设置为 2 个组件
pca = PCA(n_components=2)

# 拟合并转换数据
transformed = pca.fit_transform(X)

# 获取解释方差比
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

### StandardScaler 示例

#### Go 代码

```go
package main

import (
    "fmt"
    "github.com/yinziyang/mlkit/preprocessing/standard_scaler"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // 创建样本数据
    data := mat.NewDense(3, 2, []float64{
        0, 0,
        0, 0,
        1, 1,
    })

    // 初始化 StandardScaler
    scaler := standard_scaler.NewStandardScaler(true, true)
    
    // 拟合并转换数据
    transformed := scaler.FitTransform(data)
    
    // 保存模型
    if err := scaler.Save("scaler_model.gob"); err != nil {
        panic(err)
    }
}
```

#### Python 代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# 创建样本数据
X = np.array([
    [0, 0],
    [0, 0],
    [1, 1]
])

# 初始化 StandardScaler
scaler = StandardScaler()

# 拟合并转换数据
transformed = scaler.fit_transform(X)
```

## 依赖

- [gonum](https://github.com/gonum/gonum) - 用于矩阵运算和数值计算

## TODO

未来可能的实现包括：
- 更多预处理工具（MinMaxScaler, RobustScaler 等）
- 聚类算法（K-means, DBSCAN 等）
- 线性模型（线性回归，逻辑回归等）
- 模型评估工具
- 交叉验证工具
