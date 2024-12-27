package standard_scaler

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestStandardScaler(t *testing.T) {
	tests := []struct {
		name     string
		X        *mat.Dense
		withMean bool
		withStd  bool
		want     [][]float64
	}{
		{
			name: "Case 1: Matrix1 with_mean=false, with_std=false",
			X: mat.NewDense(3, 7, []float64{
				6, 5, 4, 3, 8, 2, 9,
				5, 1, 10, 2, 3, 8, 7,
				5, 14, 2, 3, 6, 3, 2,
			}),
			withMean: false,
			withStd:  false,
			want: [][]float64{
				{6, 5, 4, 3, 8, 2, 9},
				{5, 1, 10, 2, 3, 8, 7},
				{5, 14, 2, 3, 6, 3, 2},
			},
		},
		{
			name: "Case 2: Matrix1 with_mean=true, with_std=false",
			X: mat.NewDense(3, 7, []float64{
				6, 5, 4, 3, 8, 2, 9,
				5, 1, 10, 2, 3, 8, 7,
				5, 14, 2, 3, 6, 3, 2,
			}),
			withMean: true,
			withStd:  false,
			want: [][]float64{
				{0.66666667, -1.66666667, -1.33333333, 0.33333333, 2.33333333, -2.33333333, 3},
				{-0.33333333, -5.66666667, 4.66666667, -0.66666667, -2.66666667, 3.66666667, 1},
				{-0.33333333, 7.33333333, -3.33333333, 0.33333333, 0.33333333, -1.33333333, -4},
			},
		},
		{
			name: "Case 3: Matrix1 with_mean=false, with_std=true",
			X: mat.NewDense(3, 7, []float64{
				6, 5, 4, 3, 8, 2, 9,
				5, 1, 10, 2, 3, 8, 7,
				5, 14, 2, 3, 6, 3, 2,
			}),
			withMean: false,
			withStd:  true,
			want: [][]float64{
				{12.72792206, 0.91970901, 1.17669681, 6.36396103, 3.89331411, 0.76200076, 3.05714799},
				{10.60660172, 0.1839418, 2.94174203, 4.24264069, 1.45999279, 3.04800305, 2.37778177},
				{10.60660172, 2.57518523, 0.58834841, 6.36396103, 2.91998558, 1.14300114, 0.67936622},
			},
		},
		{
			name: "Case 4: Matrix1 with_mean=true, with_std=true",
			X: mat.NewDense(3, 7, []float64{
				6, 5, 4, 3, 8, 2, 9,
				5, 1, 10, 2, 3, 8, 7,
				5, 14, 2, 3, 6, 3, 2,
			}),
			withMean: true,
			withStd:  true,
			want: [][]float64{
				{1.41421356, -0.30656967, -0.39223227, 0.70710678, 1.13554995, -0.88900089, 1.01904933},
				{-0.70710678, -1.04233688, 1.37281295, -1.41421356, -1.29777137, 1.3970014, 0.33968311},
				{-0.70710678, 1.34890655, -0.98058068, 0.70710678, 0.16222142, -0.50800051, -1.35873244},
			},
		},
		{
			name: "Case 5: Matrix2 with_mean=false, with_std=false",
			X: mat.NewDense(4, 2, []float64{
				0, 0,
				0, 0,
				1, 1,
				1, 1,
			}),
			withMean: false,
			withStd:  false,
			want: [][]float64{
				{0, 0},
				{0, 0},
				{1, 1},
				{1, 1},
			},
		},
		{
			name: "Case 6: Matrix2 with_mean=true, with_std=false",
			X: mat.NewDense(4, 2, []float64{
				0, 0,
				0, 0,
				1, 1,
				1, 1,
			}),
			withMean: true,
			withStd:  false,
			want: [][]float64{
				{-0.5, -0.5},
				{-0.5, -0.5},
				{0.5, 0.5},
				{0.5, 0.5},
			},
		},
		{
			name: "Case 7: Matrix2 with_mean=false, with_std=true",
			X: mat.NewDense(4, 2, []float64{
				0, 0,
				0, 0,
				1, 1,
				1, 1,
			}),
			withMean: false,
			withStd:  true,
			want: [][]float64{
				{0, 0},
				{0, 0},
				{2, 2},
				{2, 2},
			},
		},
		{
			name: "Case 8: Matrix2 with_mean=true, with_std=true",
			X: mat.NewDense(4, 2, []float64{
				0, 0,
				0, 0,
				1, 1,
				1, 1,
			}),
			withMean: true,
			withStd:  true,
			want: [][]float64{
				{-1, -1},
				{-1, -1},
				{1, 1},
				{1, 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scaler := NewStandardScaler(tt.withMean, tt.withStd)
			got, err := scaler.FitTransform(tt.X)
			if err != nil {
				t.Errorf("StandardScaler.FitTransform() error = %v", err)
				return
			}

			rows, cols := got.Dims()
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if !almostEqual(got.At(i, j), tt.want[i][j], 1e-8) {
						t.Errorf("StandardScaler.FitTransform() at [%d,%d] = %v, want %v",
							i, j, got.At(i, j), tt.want[i][j])
					}
				}
			}

			// 测试 InverseTransform
			inverse, err := scaler.InverseTransform(got)
			if err != nil {
				t.Errorf("StandardScaler.InverseTransform() error = %v", err)
				return
			}

			// 验证逆变换后的结果是否与原始数据相同
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if !almostEqual(inverse.At(i, j), tt.X.At(i, j), 1e-8) {
						t.Errorf("StandardScaler.InverseTransform() at [%d,%d] = %v, want %v",
							i, j, inverse.At(i, j), tt.X.At(i, j))
					}
				}
			}
		})
	}
}

// almostEqual 检查两个浮点数是否在给定的误差范围内相等
func almostEqual(a, b float64, epsilon float64) bool {
	return math.Abs(a-b) <= epsilon
}
