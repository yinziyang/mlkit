package pca

import (
	"math"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPCA(t *testing.T) {
	// 测试数据
	X := mat.NewDense(3, 7, []float64{
		6, 5, 4, 3, 8, 2, 9,
		5, 1, 10, 2, 3, 8, 7,
		5, 14, 2, 3, 6, 3, 2,
	})

	tests := []struct {
		name                  string
		numComponents         int
		expectedTransformed   [][]float64
		expectedReconstructed [][]float64
	}{
		{
			name:          "Full components",
			numComponents: 0,
			expectedTransformed: [][]float64{
				{-0.54773351, 4.96990825, 5.76839921e-16},
				{-8.28142697, -2.72359455, 5.76839921e-16},
				{8.82916048, -2.24631370, 5.76839921e-16},
			},
			expectedReconstructed: [][]float64{
				{6, 5, 4, 3, 8, 2, 9},
				{5, 1, 10, 2, 3, 8, 7},
				{5, 14, 2, 3, 6, 3, 2},
			},
		},
		{
			name:          "Two components",
			numComponents: 2,
			expectedTransformed: [][]float64{
				{-0.54773351, 4.96990825},
				{-8.28142697, -2.72359455},
				{8.82916048, -2.24631370},
			},
			expectedReconstructed: [][]float64{
				{6, 5, 4, 3, 8, 2, 9},
				{5, 1, 10, 2, 3, 8, 7},
				{5, 14, 2, 3, 6, 3, 2},
			},
		},
		{
			name:          "One component",
			numComponents: 1,
			expectedTransformed: [][]float64{
				{-0.54773351},
				{-8.28142697},
				{8.82916048},
			},
			expectedReconstructed: [][]float64{
				{5.33537651, 6.24668619, 5.58455316, 2.63577498, 5.57807797, 4.48574858, 6.16876065},
				{5.36422502, 0.31679469, 9.13163782, 2.19960153, 4.32725461, 6.63776735, 8.55156749},
				{5.30039847, 13.43651912, 1.28380902, 3.16462349, 7.09466742, 1.87648407, 3.27967186},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pca := NewPCA(tt.numComponents)
			transformed := pca.FitTransform(X)

			// 检查转换后的数据
			rows, cols := transformed.Dims()
			if rows != len(tt.expectedTransformed) || cols != len(tt.expectedTransformed[0]) {
				t.Errorf("Transformed dimensions mismatch: got (%d,%d), want (%d,%d)",
					rows, cols, len(tt.expectedTransformed), len(tt.expectedTransformed[0]))
			}

			// 验证转换后的数据值
			tolerance := 1e-6
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if math.Abs(transformed.At(i, j)-tt.expectedTransformed[i][j]) > tolerance {
						t.Errorf("Transformed data mismatch at (%d,%d): got %v, want %v",
							i, j, transformed.At(i, j), tt.expectedTransformed[i][j])
					}
				}
			}

			// 检查重构数据
			reconstructed := pca.InverseTransform(transformed)
			rows, cols = reconstructed.Dims()
			if rows != len(tt.expectedReconstructed) || cols != len(tt.expectedReconstructed[0]) {
				t.Errorf("Reconstructed dimensions mismatch: got (%d,%d), want (%d,%d)",
					rows, cols, len(tt.expectedReconstructed), len(tt.expectedReconstructed[0]))
			}

			// 验证重构数据值
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if math.Abs(reconstructed.At(i, j)-tt.expectedReconstructed[i][j]) > tolerance {
						t.Errorf("Reconstructed data mismatch at (%d,%d): got %v, want %v",
							i, j, reconstructed.At(i, j), tt.expectedReconstructed[i][j])
					}
				}
			}
		})
	}
}

func TestPCASaveLoad(t *testing.T) {
	// 创建一个临时文件
	tmpfile, err := os.CreateTemp("", "pca_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	// 准备测试数据
	X := mat.NewDense(3, 7, []float64{
		6, 5, 4, 3, 8, 2, 9,
		5, 1, 10, 2, 3, 8, 7,
		5, 14, 2, 3, 6, 3, 2,
	})

	// 训练原始模型
	originalPCA := NewPCA(2)
	originalTransformed := originalPCA.FitTransform(X)

	// 保存模型
	if err := originalPCA.Save(tmpfile.Name()); err != nil {
		t.Fatalf("Failed to save PCA model: %v", err)
	}

	// 创建新的PCA模型并加载保存的数据
	loadedPCA := NewPCA(2)
	if err := loadedPCA.Load(tmpfile.Name()); err != nil {
		t.Fatalf("Failed to load PCA model: %v", err)
	}

	// 使用加载的模型转换数据
	loadedTransformed := loadedPCA.Transform(X)

	// 比较原始模型和加载模型的结果
	originalRows, originalCols := originalTransformed.Dims()
	loadedRows, loadedCols := loadedTransformed.Dims()

	if originalRows != loadedRows || originalCols != loadedCols {
		t.Errorf("Dimension mismatch: original=(%d,%d), loaded=(%d,%d)",
			originalRows, originalCols, loadedRows, loadedCols)
	}

	tolerance := 1e-10
	for i := 0; i < originalRows; i++ {
		for j := 0; j < originalCols; j++ {
			if math.Abs(originalTransformed.At(i, j)-loadedTransformed.At(i, j)) > tolerance {
				t.Errorf("Value mismatch at (%d,%d): original=%v, loaded=%v",
					i, j, originalTransformed.At(i, j), loadedTransformed.At(i, j))
			}
		}
	}

	// 检查其他属性是否正确加载
	if len(originalPCA.varianceRatio) != len(loadedPCA.varianceRatio) {
		t.Error("varianceRatio length mismatch")
	}

	for i := range originalPCA.varianceRatio {
		if math.Abs(originalPCA.varianceRatio[i]-loadedPCA.varianceRatio[i]) > tolerance {
			t.Errorf("varianceRatio mismatch at %d: original=%v, loaded=%v",
				i, originalPCA.varianceRatio[i], loadedPCA.varianceRatio[i])
		}
	}
}
