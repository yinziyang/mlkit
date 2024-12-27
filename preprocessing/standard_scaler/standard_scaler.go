package standard_scaler

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

// StandardScaler implements standardization by centering and scaling the data
type StandardScaler struct {
	Mean_    []float64
	Scale_   []float64
	fitted   bool
	WithMean bool
	WithStd  bool
}

// NewStandardScaler creates a new StandardScaler instance
func NewStandardScaler(withMean bool, withStd bool) *StandardScaler {
	return &StandardScaler{
		fitted:   false,
		WithMean: withMean,
		WithStd:  withStd,
	}
}

// Fit computes the mean and standard deviation of X for later scaling
func (s *StandardScaler) Fit(X mat.Matrix) error {
	if X == nil {
		return fmt.Errorf("input matrix X cannot be nil")
	}

	rows, cols := X.Dims()
	if rows == 0 || cols == 0 {
		return fmt.Errorf("input matrix X cannot be empty")
	}

	s.Mean_ = make([]float64, cols)
	s.Scale_ = make([]float64, cols)

	// 计算每个特征的均值
	if s.WithMean || s.WithStd {
		for j := 0; j < cols; j++ {
			sum := 0.0
			for i := 0; i < rows; i++ {
				sum += X.At(i, j)
			}
			s.Mean_[j] = sum / float64(rows)
		}
	}

	// 计算每个特征的标准差
	if s.WithStd {
		for j := 0; j < cols; j++ {
			sumSquared := 0.0
			for i := 0; i < rows; i++ {
				diff := X.At(i, j) - s.Mean_[j]
				sumSquared += diff * diff
			}
			// 使用有偏估计 (ddof=0)
			s.Scale_[j] = math.Sqrt(sumSquared / float64(rows))
			if s.Scale_[j] == 0 {
				s.Scale_[j] = 1.0
			}
		}
	} else {
		for j := 0; j < cols; j++ {
			s.Scale_[j] = 1.0
		}
	}

	s.fitted = true
	return nil
}

// Transform standardizes X using the mean and standard deviation learned in Fit
func (s *StandardScaler) Transform(X mat.Matrix) (mat.Matrix, error) {
	if !s.fitted {
		return nil, fmt.Errorf("StandardScaler must be fitted before transform")
	}

	rows, cols := X.Dims()
	if cols != len(s.Mean_) {
		return nil, fmt.Errorf("x has %d features, but StandardScaler was fitted with %d features", cols, len(s.Mean_))
	}

	scaled := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := X.At(i, j)
			if s.WithMean {
				value -= s.Mean_[j]
			}
			if s.WithStd {
				value /= s.Scale_[j]
			}
			scaled.Set(i, j, value)
		}
	}

	return scaled, nil
}

// FitTransform combines Fit and Transform operations
func (s *StandardScaler) FitTransform(X mat.Matrix) (mat.Matrix, error) {
	err := s.Fit(X)
	if err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// InverseTransform scales back the data to the original representation
func (s *StandardScaler) InverseTransform(X mat.Matrix) (mat.Matrix, error) {
	if !s.fitted {
		return nil, fmt.Errorf("StandardScaler must be fitted before inverse_transform")
	}

	rows, cols := X.Dims()
	if cols != len(s.Mean_) {
		return nil, fmt.Errorf("x has %d features, but StandardScaler was fitted with %d features", cols, len(s.Mean_))
	}

	inverse := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := X.At(i, j)
			if s.WithStd {
				value *= s.Scale_[j]
			}
			if s.WithMean {
				value += s.Mean_[j]
			}
			inverse.Set(i, j, value)
		}
	}

	return inverse, nil
}

// Save saves the StandardScaler model to a file
func (s *StandardScaler) Save(filename string) error {
	if !s.fitted {
		return fmt.Errorf("StandardScaler must be fitted before saving")
	}

	model := struct {
		Mean_    []float64 `json:"mean"`
		Scale_   []float64 `json:"scale"`
		WithMean bool      `json:"with_mean"`
		WithStd  bool      `json:"with_std"`
	}{
		Mean_:    s.Mean_,
		Scale_:   s.Scale_,
		WithMean: s.WithMean,
		WithStd:  s.WithStd,
	}

	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling model: %v", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("error writing model to file: %v", err)
	}

	return nil
}

// Load loads the StandardScaler model from a file
func Load(filename string) (*StandardScaler, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading model file: %v", err)
	}

	var model struct {
		Mean_    []float64 `json:"mean"`
		Scale_   []float64 `json:"scale"`
		WithMean bool      `json:"with_mean"`
		WithStd  bool      `json:"with_std"`
	}

	err = json.Unmarshal(data, &model)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling model: %v", err)
	}

	scaler := &StandardScaler{
		Mean_:    model.Mean_,
		Scale_:   model.Scale_,
		WithMean: model.WithMean,
		WithStd:  model.WithStd,
		fitted:   true,
	}

	return scaler, nil
}
