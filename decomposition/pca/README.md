# Principal Component Analysis (PCA)

这是一个基于SVD（奇异值分解）的主成分分析实现，完全兼容sklearn.decomposition.PCA的API设计。

## 功能特点

- 支持指定保留的主成分数量
- 提供每个主成分的解释方差比
- 支持数据的逆变换（还原）
- 使用SVD进行主成分计算
- 提供详细的模型属性（Components_, Mean_, SingularValues_等）

## 使用示例

```go
import (
    "github.com/yinziyang/mlkit/decomposition/pca"
    "gonum.org/v1/gonum/mat"
)

// 创建数据矩阵
X := mat.NewDense(100, 4, data) // 100个样本，4个特征

// 创建PCA模型，保留2个主成分
pca := pca.NewPCA(2)

// 拟合数据并转换
reduced, err := pca.FitTransform(X)
if err != nil {
    // 处理错误
}

// 查看解释方差比
fmt.Printf("Explained variance ratio: %v\n", pca.ExplainedVarianceRatio_)
fmt.Printf("Total explained variance ratio: %v\n", pca.TotalExplainedVarianceRatio_)

// 还原数据
restored, err := pca.InverseTransform(reduced)
if err != nil {
    // 处理错误
}
```

## 主要方法

### NewPCA

```go
func NewPCA(numComponents int) *PCA
```

创建新的PCA实例。参数：
- numComponents: 要保留的主成分数量。如果为0，则保留所有主成分。

### Fit

```go
func (p *PCA) Fit(X *mat.Dense) (*PCA, error)
```

使用训练数据拟合PCA模型。

### Transform

```go
func (p *PCA) Transform(X *mat.Dense) (*mat.Dense, error)
```

将数据转换到主成分空间。

### FitTransform

```go
func (p *PCA) FitTransform(X *mat.Dense) (*mat.Dense, error)
```

拟合PCA模型并转换数据。

### InverseTransform

```go
func (p *PCA) InverseTransform(X *mat.Dense) (*mat.Dense, error)
```

将降维后的数据转换回原始特征空间。

## 模型属性

- Components_: 主成分（特征向量）
- Mean_: 训练数据的特征均值
- SingularValues_: SVD分解得到的奇异值
- ExplainedVarianceRatio_: 每个主成分解释的方差比例
- TotalExplainedVarianceRatio_: 保留的主成分解释的总方差比例

## 注意事项

1. 输入数据必须是`*mat.Dense`类型
2. 在使用Transform之前必须先调用Fit方法
3. 数据会自动进行中心化处理
4. NumComponents不能大于特征数量
5. 使用SVD分解可能在处理大规模数据时较慢

## 依赖

- [gonum/mat](https://pkg.go.dev/gonum.org/v1/gonum/mat) - 用于矩阵运算

## Installation

```bash
go get github.com/yinziyang/mlkit/decomposition/pca
```

## Usage

### Basic Usage

```go
package main

import (
    "fmt"
    "github.com/yinziyang/mlkit/decomposition/pca"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Create sample data (3 samples, 7 features)
    data := mat.NewDense(3, 7, []float64{
        6, 5, 4, 3, 8, 2, 9,
        5, 1, 10, 2, 3, 8, 7,
        5, 14, 2, 3, 6, 3, 2,
    })

    // Initialize PCA with 2 components
    pca := pca.NewPCA(2)
    
    // Fit and transform data
    transformed := pca.FitTransform(data)
    
    // Get explained variance ratio
    fmt.Printf("Explained variance ratio: %v\n", pca.ExplainedVarianceRatio())
}
```

### Model Persistence

```go
// Save model to file
if err := pca.Save("pca_model.gob"); err != nil {
    panic(err)
}

// Load model from file
newPCA := pca.NewPCA(0)
if err := newPCA.Load("pca_model.gob"); err != nil {
    panic(err)
}

// Use loaded model
transformed := newPCA.Transform(newData)
```

## API Reference

### NewPCA

```go
func NewPCA(numComponents int) *PCA
```
Creates a new PCA instance. If numComponents is 0 or negative, it will use all components.

### FitTransform

```go
func (pca *PCA) FitTransform(X *mat.Dense) *mat.Dense
```
Fits the model with X and applies the dimensionality reduction on X.

### Transform

```go
func (pca *PCA) Transform(X *mat.Dense) *mat.Dense
```
Applies dimensionality reduction to X.

### InverseTransform

```go
func (pca *PCA) InverseTransform(X *mat.Dense) *mat.Dense
```
Transform data back to its original space.

### Save/Load

```go
func (pca *PCA) Save(filename string) error
func (pca *PCA) Load(filename string) error
```
Save and load PCA model to/from a file.

## Implementation Details

The implementation follows these steps:

1. Center the data by subtracting the mean
2. Compute the SVD (Singular Value Decomposition)
3. Calculate explained variance ratio
4. Project the data onto principal components

## Performance

The implementation uses gonum's efficient matrix operations and SVD computation. Memory usage is optimized by avoiding unnecessary matrix copies.

## Testing

The package includes comprehensive tests that verify:
- Correctness of PCA transformation
- Compatibility with scikit-learn's output
- Model persistence functionality
- Edge cases and error handling

## Dependencies

- [gonum](https://github.com/gonum/gonum) - For matrix operations and SVD computation
