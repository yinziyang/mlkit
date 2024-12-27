package pca

import (
	"encoding/gob"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

type PCA struct {
	NumComponents int
	svd          *mat.SVD
	meanVec      *mat.Dense
	components   *mat.Dense
	varianceRatio []float64
	isFitted     bool
}

// NewPCA initializes a PCA instance with the specified number of components.
// If numComponents is 0, it defaults to using all features.
func NewPCA(numComponents int) *PCA {
	return &PCA{NumComponents: numComponents, isFitted: false}
}

// FitTransform fits the PCA model and transforms the data.
func (pca *PCA) FitTransform(X mat.Matrix) *mat.Dense {
	return pca.Fit(X).Transform(X)
}

// Fit computes the PCA model using the input data.
func (pca *PCA) Fit(X mat.Matrix) *PCA {
	if pca.NumComponents < 0 {
		panic("Number of components cannot be less than zero")
	}

	rows, cols := X.Dims()
	if rows < 1 || cols < 1 {
		panic("Empty matrix")
	}

	// Compute mean and center the data
	pca.meanVec = mean(X)
	centeredX := matrixSubVector(X, pca.meanVec)

	// Perform SVD decomposition
	pca.svd = &mat.SVD{}
	ok := pca.svd.Factorize(centeredX, mat.SVDThin)
	if !ok {
		panic("Unable to factorize matrix")
	}

	// Get singular values and compute variance ratios
	singularValues := pca.svd.Values(nil)
	totalVariance := 0.0
	variances := make([]float64, len(singularValues))
	for i, s := range singularValues {
		variances[i] = s * s
		totalVariance += variances[i]
	}

	pca.varianceRatio = make([]float64, len(singularValues))
	for i := range singularValues {
		pca.varianceRatio[i] = variances[i] / totalVariance
	}

	// Determine number of components
	numComponents := pca.NumComponents
	if numComponents == 0 {
		numComponents = min(rows, cols)
	} else if numComponents > min(rows, cols) {
		numComponents = min(rows, cols)
	}

	// Get the right singular vectors (V matrix)
	vTemp := new(mat.Dense)
	pca.svd.VTo(vTemp)

	// Store components
	pca.components = mat.NewDense(numComponents, cols, nil)
	for i := 0; i < numComponents; i++ {
		for j := 0; j < cols; j++ {
			pca.components.Set(i, j, vTemp.At(j, i))
		}
	}

	pca.isFitted = true

	return pca
}

// Transform applies the fitted PCA model to the input data.
func (pca *PCA) Transform(X mat.Matrix) *mat.Dense {
	if !pca.isFitted {
		panic("PCA model must be fitted before calling Transform")
	}

	_, cols := X.Dims()
	_, componentCols := pca.components.Dims()
	if cols != componentCols {
		panic(fmt.Sprintf("Input matrix has %d features but model was trained with %d features", 
			cols, componentCols))
	}

	// Center the data
	centeredX := matrixSubVector(X, pca.meanVec)

	// Project data onto principal components
	var transformed mat.Dense
	transformed.Mul(centeredX, pca.components.T())

	return &transformed
}

// InverseTransform transforms data back to its original space.
func (pca *PCA) InverseTransform(X mat.Matrix) *mat.Dense {
	if !pca.isFitted {
		panic("PCA model must be fitted before calling InverseTransform")
	}

	_, cols := X.Dims()
	componentRows, componentCols := pca.components.Dims()
	if cols != componentRows {
		panic(fmt.Sprintf("Input matrix has %d features but model has %d components", 
			cols, componentRows))
	}

	// Project back to original space
	var reconstructed mat.Dense
	reconstructed.Mul(X, pca.components)

	// Add mean back
	rows, _ := X.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < componentCols; j++ {
			reconstructed.Set(i, j, reconstructed.At(i, j)+pca.meanVec.At(0, j))
		}
	}

	return &reconstructed
}

// ExplainedVarianceRatio returns the explained variance ratio for each component
func (pca *PCA) ExplainedVarianceRatio() []float64 {
	if pca.varianceRatio == nil {
		return nil
	}
	ratio := make([]float64, len(pca.varianceRatio))
	copy(ratio, pca.varianceRatio)
	return ratio
}

// TotalExplainedVarianceRatio returns the sum of explained variance ratios
func (pca *PCA) TotalExplainedVarianceRatio() float64 {
	if pca.varianceRatio == nil {
		return 0
	}
	total := 0.0
	numComponents := pca.NumComponents
	if numComponents == 0 {
		numComponents = len(pca.varianceRatio)
	}
	for i := 0; i < numComponents && i < len(pca.varianceRatio); i++ {
		total += pca.varianceRatio[i]
	}
	return total
}

// Components returns the principal components
func (pca *PCA) Components() *mat.Dense {
	if pca.components == nil {
		return nil
	}
	r, c := pca.components.Dims()
	result := mat.NewDense(r, c, nil)
	result.Copy(pca.components)
	return result
}

// Mean returns the mean vector
func (pca *PCA) Mean() *mat.Dense {
	if pca.meanVec == nil {
		return nil
	}
	r, c := pca.meanVec.Dims()
	result := mat.NewDense(r, c, nil)
	result.Copy(pca.meanVec)
	return result
}

// Save 将PCA模型保存到文件
func (pca *PCA) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// 创建一个编码器
	encoder := gob.NewEncoder(file)

	// 创建一个包含所有需要保存的数据的结构
	data := struct {
		NumComponents     int
		Mean             []float64
		Components       []float64
		VarianceRatio    []float64
		ComponentsShape  [2]int
		IsFitted        bool
	}{
		NumComponents:  pca.NumComponents,
		VarianceRatio: pca.varianceRatio,
		IsFitted:      pca.isFitted,
	}

	// 将矩阵数据转换为切片
	if pca.meanVec != nil {
		r, c := pca.meanVec.Dims()
		data.Mean = make([]float64, r*c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				data.Mean[i*c+j] = pca.meanVec.At(i, j)
			}
		}
	}

	if pca.components != nil {
		r, c := pca.components.Dims()
		data.Components = make([]float64, r*c)
		data.ComponentsShape = [2]int{r, c}
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				data.Components[i*c+j] = pca.components.At(i, j)
			}
		}
	}

	// 编码并保存数据
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("failed to encode PCA model: %v", err)
	}

	return nil
}

// Load 从文件加载PCA模型
func (pca *PCA) Load(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// 创建一个解码器
	decoder := gob.NewDecoder(file)

	// 创建一个临时结构来接收数据
	var data struct {
		NumComponents     int
		Mean             []float64
		Components       []float64
		VarianceRatio    []float64
		ComponentsShape  [2]int
		IsFitted        bool
	}

	// 解码数据
	if err := decoder.Decode(&data); err != nil {
		return fmt.Errorf("failed to decode PCA model: %v", err)
	}

	// 恢复PCA模型的状态
	pca.NumComponents = data.NumComponents
	pca.varianceRatio = data.VarianceRatio
	pca.isFitted = data.IsFitted

	// 恢复均值矩阵
	if len(data.Mean) > 0 {
		pca.meanVec = mat.NewDense(1, len(data.Mean), data.Mean)
	}

	// 恢复组件矩阵
	if len(data.Components) > 0 {
		pca.components = mat.NewDense(data.ComponentsShape[0], data.ComponentsShape[1], data.Components)
	}

	return nil
}

// Helper functions

// mean computes the mean of the columns of the input matrix.
func mean(matrix mat.Matrix) *mat.Dense {
	rows, cols := matrix.Dims()
	meanVector := make([]float64, cols)
	for j := 0; j < cols; j++ {
		col := make([]float64, rows)
		for i := 0; i < rows; i++ {
			col[i] = matrix.At(i, j)
		}
		sum := 0.0
		for _, val := range col {
			sum += val
		}
		meanVector[j] = sum / float64(rows)
	}
	return mat.NewDense(1, cols, meanVector)
}

// matrixSubVector subtracts a vector from each row of the input matrix.
func matrixSubVector(m mat.Matrix, vec *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	_, vecCols := vec.Dims()
	if cols != vecCols {
		panic(fmt.Sprintf("Dimension mismatch: matrix has %d columns but vector has %d", cols, vecCols))
	}

	result := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, m.At(i, j)-vec.At(0, j))
		}
	}
	return result
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func init() {
	gob.Register(&mat.Dense{})
}
