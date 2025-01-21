package matrix

import (
	"testing"
)

func TestSparseMatrix_MergeRows(t *testing.T) {
	tests := []struct {
		name    string
		matrix1 *SparseMatrix
		matrix2 *SparseMatrix
		want    *SparseMatrix
		wantErr bool
	}{
		{
			name: "正常合并两个矩阵的行",
			matrix1: &SparseMatrix{
				Rows:   2,
				Cols:   3,
				Data:   []float64{1, 2},
				RowIdx: []int{0, 1},
				ColIdx: []int{0, 1},
			},
			matrix2: &SparseMatrix{
				Rows:   2,
				Cols:   3,
				Data:   []float64{3, 4},
				RowIdx: []int{0, 1},
				ColIdx: []int{1, 2},
			},
			want: &SparseMatrix{
				Rows:   4,
				Cols:   3,
				Data:   []float64{1, 2, 3, 4},
				RowIdx: []int{0, 1, 2, 3},
				ColIdx: []int{0, 1, 1, 2},
			},
			wantErr: false,
		},
		{
			name: "列数不匹配",
			matrix1: &SparseMatrix{
				Rows: 2,
				Cols: 3,
			},
			matrix2: &SparseMatrix{
				Rows: 2,
				Cols: 4,
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.matrix1.MergeRows(tt.matrix2)
			if (err != nil) != tt.wantErr {
				t.Errorf("MergeRows() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if tt.matrix1.Rows != tt.want.Rows {
					t.Errorf("MergeRows() Rows = %v, want %v", tt.matrix1.Rows, tt.want.Rows)
				}
				if tt.matrix1.Cols != tt.want.Cols {
					t.Errorf("MergeRows() Cols = %v, want %v", tt.matrix1.Cols, tt.want.Cols)
				}
				if len(tt.matrix1.Data) != len(tt.want.Data) {
					t.Errorf("MergeRows() Data length = %v, want %v", len(tt.matrix1.Data), len(tt.want.Data))
				}
				// 检查数据内容
				for i := range tt.want.Data {
					if tt.matrix1.Data[i] != tt.want.Data[i] {
						t.Errorf("MergeRows() Data[%d] = %v, want %v", i, tt.matrix1.Data[i], tt.want.Data[i])
					}
				}
			}
		})
	}
}

func TestSparseMatrix_MergeCols(t *testing.T) {
	tests := []struct {
		name    string
		matrix1 *SparseMatrix
		matrix2 *SparseMatrix
		want    *SparseMatrix
		wantErr bool
	}{
		{
			name: "正常合并两个矩阵的列",
			matrix1: &SparseMatrix{
				Rows:   2,
				Cols:   2,
				Data:   []float64{1, 2},
				RowIdx: []int{0, 1},
				ColIdx: []int{0, 1},
			},
			matrix2: &SparseMatrix{
				Rows:   2,
				Cols:   2,
				Data:   []float64{3, 4},
				RowIdx: []int{0, 1},
				ColIdx: []int{0, 1},
			},
			want: &SparseMatrix{
				Rows:   2,
				Cols:   4,
				Data:   []float64{1, 2, 3, 4},
				RowIdx: []int{0, 1, 0, 1},
				ColIdx: []int{0, 1, 2, 3},
			},
			wantErr: false,
		},
		{
			name: "行数不匹配",
			matrix1: &SparseMatrix{
				Rows: 2,
				Cols: 2,
			},
			matrix2: &SparseMatrix{
				Rows: 3,
				Cols: 2,
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.matrix1.MergeCols(tt.matrix2)
			if (err != nil) != tt.wantErr {
				t.Errorf("MergeCols() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if tt.matrix1.Rows != tt.want.Rows {
					t.Errorf("MergeCols() Rows = %v, want %v", tt.matrix1.Rows, tt.want.Rows)
				}
				if tt.matrix1.Cols != tt.want.Cols {
					t.Errorf("MergeCols() Cols = %v, want %v", tt.matrix1.Cols, tt.want.Cols)
				}
				if len(tt.matrix1.Data) != len(tt.want.Data) {
					t.Errorf("MergeCols() Data length = %v, want %v", len(tt.matrix1.Data), len(tt.want.Data))
				}
				// 检查数据内容
				for i := range tt.want.Data {
					if tt.matrix1.Data[i] != tt.want.Data[i] {
						t.Errorf("MergeCols() Data[%d] = %v, want %v", i, tt.matrix1.Data[i], tt.want.Data[i])
					}
				}
				// 检查行列索引
				for i := range tt.want.RowIdx {
					if tt.matrix1.RowIdx[i] != tt.want.RowIdx[i] {
						t.Errorf("MergeCols() RowIdx[%d] = %v, want %v", i, tt.matrix1.RowIdx[i], tt.want.RowIdx[i])
					}
					if tt.matrix1.ColIdx[i] != tt.want.ColIdx[i] {
						t.Errorf("MergeCols() ColIdx[%d] = %v, want %v", i, tt.matrix1.ColIdx[i], tt.want.ColIdx[i])
					}
				}
			}
		})
	}
}
