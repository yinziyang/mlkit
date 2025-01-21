// Package matrix 提供矩阵运算相关的功能实现
package matrix

import (
	"fmt"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// SparseMatrix 表示稀疏矩阵的数据结构
// 采用CSR(Compressed Sparse Row)格式存储，只存储非零元素
type SparseMatrix struct {
	Rows   int       // 矩阵的行数
	Cols   int       // 矩阵的列数
	Data   []float64 // 非零元素值
	RowIdx []int     // 非零元素的行索引
	ColIdx []int     // 非零元素的列索引
}

// GetNumFeatures 返回特征维度（列数）
func (s *SparseMatrix) GetNumFeatures() int {
	return s.Cols
}

// Validate 验证稀疏矩阵的有效性
// 检查所有列索引是否在合法范围内
func (sm *SparseMatrix) Validate() error {
	// 检查列索引是否超出特征维度
	for _, col := range sm.ColIdx {
		if col > sm.Cols {
			return fmt.Errorf("列索引 %d 超出特征维度 %d", col, sm.Cols)
		}
	}
	return nil
}

// ToGonumDense 将稀疏矩阵转换为Gonum库的Dense矩阵格式
func (sm *SparseMatrix) ToGonumDense() (*mat.Dense, error) {
	if err := sm.Validate(); err != nil {
		return nil, err
	}

	sparse := mat.NewDense(sm.Rows, sm.Cols, nil)

	for i := 0; i < len(sm.Data); i++ {
		sparse.Set(sm.RowIdx[i], sm.ColIdx[i], sm.Data[i])
	}

	return sparse, nil
}

// ToDense 将稀疏矩阵转换为普通的二维密集矩阵
func (sm *SparseMatrix) ToDense() [][]float32 {
	// 初始化稠密矩阵
	dense := make([][]float32, sm.Rows)
	for i := range dense {
		dense[i] = make([]float32, sm.Cols)
	}

	// 填充稠密矩阵
	for i := 0; i < len(sm.Data); i++ {
		row := sm.RowIdx[i]
		col := sm.ColIdx[i]
		dense[row][col] = float32(sm.Data[i])
	}

	return dense
}

// Binarize 将矩阵二值化
// 所有非零元素的值都设置为1
func (sm *SparseMatrix) Binarize() {
	for i := 0; i < len(sm.Data); i++ {
		sm.Data[i] = 1 // 将所有非零元素的值设置为 1
	}
}

// AddConstantFeature 为矩阵添加一个常数特征列
// 在矩阵最后添加一列，其值都为指定的常数
func (sm *SparseMatrix) AddConstantFeature(constant float64) {
	// 增加一列新的特征
	sm.Cols++ // 列数也增加

	// 为每一行添加新的常数特征列
	sm.Data = append(sm.Data, make([]float64, sm.Rows)...)
	sm.RowIdx = append(sm.RowIdx, make([]int, sm.Rows)...)
	sm.ColIdx = append(sm.ColIdx, make([]int, sm.Rows)...)

	// 将数据填充到新添加的常数特征列
	dataIdx := len(sm.Data) - sm.Rows // 新的特征列的起始位置
	for i := 0; i < sm.Rows; i++ {
		sm.Data[dataIdx+i] = constant
		sm.RowIdx[dataIdx+i] = i           // 行索引
		sm.ColIdx[dataIdx+i] = sm.Cols - 1 // 新特征的列索引
	}
}

// String 实现Stringer接口，提供矩阵的字符串表示
// 按行列顺序输出所有非零元素
func (s *SparseMatrix) String() string {
	// 创建entry结构来存储矩阵元素
	type entry struct {
		row   int
		col   int
		value float64
	}

	// 收集所有非零元素
	entries := make([]entry, len(s.Data))
	for i := range s.Data {
		entries[i] = entry{
			row:   s.RowIdx[i],
			col:   s.ColIdx[i],
			value: s.Data[i],
		}
	}

	// 按行索引和列索引排序
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].row != entries[j].row {
			return entries[i].row < entries[j].row
		}
		return entries[i].col < entries[j].col
	})

	// 构建输出字符串
	var result strings.Builder
	result.WriteString(fmt.Sprintf("numFeatures: %d\n", s.Cols))
	for _, e := range entries {
		result.WriteString(fmt.Sprintf("  (%d, %d)\t%g\n",
			e.row, e.col, e.value))
	}
	return result.String()
}

// MergeRows 合并多个稀疏矩阵的行
// 要求所有矩阵的列数（特征维度）相同
func (sm *SparseMatrix) MergeRows(other ...*SparseMatrix) error {
	// 验证所有矩阵的特征维度是否匹配
	for _, m := range other {
		if sm.Cols != m.Cols {
			return fmt.Errorf("特征维度不匹配: %d != %d", sm.Cols, m.Cols)
		}
	}

	// 计算总数据量
	totalData := len(sm.Data)
	totalRows := sm.Rows
	for _, m := range other {
		totalData += len(m.Data)
		totalRows += m.Rows
	}

	// 创建新的存储空间
	newData := make([]float64, 0, totalData)
	newRowIdx := make([]int, 0, totalData)
	newColIdx := make([]int, 0, totalData)

	// 复制第一个矩阵的数据
	newData = append(newData, sm.Data...)
	newColIdx = append(newColIdx, sm.ColIdx...)
	newRowIdx = append(newRowIdx, sm.RowIdx...)

	// 复制其他矩阵的数据，并调整行索引
	currentRows := sm.Rows
	for _, m := range other {
		for i := 0; i < len(m.Data); i++ {
			newData = append(newData, m.Data[i])
			newColIdx = append(newColIdx, m.ColIdx[i])
			newRowIdx = append(newRowIdx, m.RowIdx[i]+currentRows)
		}
		currentRows += m.Rows
	}

	// 更新原矩阵
	sm.Rows = totalRows
	sm.Data = newData
	sm.RowIdx = newRowIdx
	sm.ColIdx = newColIdx

	return nil
}

// MergeCols 合并多个稀疏矩阵的列
// 要求所有矩阵的行数相同
func (sm *SparseMatrix) MergeCols(other ...*SparseMatrix) error {
	// 验证所有矩阵的行数是否匹配
	for _, m := range other {
		if sm.Rows != m.Rows {
			return fmt.Errorf("行数不匹配: %d != %d", sm.Rows, m.Rows)
		}
	}

	// 计算总数据量和总列数
	totalData := len(sm.Data)
	totalCols := sm.Cols
	for _, m := range other {
		totalData += len(m.Data)
		totalCols += m.Cols
	}

	// 创建新的存储空间
	newData := make([]float64, 0, totalData)
	newRowIdx := make([]int, 0, totalData)
	newColIdx := make([]int, 0, totalData)

	// 复制第一个矩阵的数据
	newData = append(newData, sm.Data...)
	newColIdx = append(newColIdx, sm.ColIdx...)
	newRowIdx = append(newRowIdx, sm.RowIdx...)

	// 复制其他矩阵的数据，并调整列索引
	currentCols := sm.Cols
	for _, m := range other {
		for i := 0; i < len(m.Data); i++ {
			newData = append(newData, m.Data[i])
			newColIdx = append(newColIdx, m.ColIdx[i]+currentCols)
			newRowIdx = append(newRowIdx, m.RowIdx[i])
		}
		currentCols += m.Cols
	}

	// 更新原矩阵
	sm.Cols = totalCols
	sm.Data = newData
	sm.RowIdx = newRowIdx
	sm.ColIdx = newColIdx

	return nil
}

func (sm *SparseMatrix) ConvertToLibSVM(labels []int, zeroBase bool, precision int) (string, error) {
	// 验证标签数量
	if len(labels) != sm.Rows {
		return "", fmt.Errorf("标签数量(%d)与矩阵行数(%d)不匹配", len(labels), sm.Rows)
	}

	// 验证精度参数
	if precision < 0 {
		precision = 6 // 默认保留6位小数
	}

	// 创建用于排序的entry结构
	type entry struct {
		row   int
		col   int
		value float64
	}

	// 收集所有非零元素并按行分组
	rowEntries := make(map[int][]entry)
	for i := range sm.Data {
		row := sm.RowIdx[i]
		rowEntries[row] = append(rowEntries[row], entry{
			row:   row,
			col:   sm.ColIdx[i],
			value: sm.Data[i],
		})
	}

	// 构建LibSVM格式字符串
	var sb strings.Builder
	for i := 0; i < sm.Rows; i++ {
		// 添加标签
		sb.WriteString(fmt.Sprintf("%d", labels[i]))

		// 获取当前行的所有特征
		entries := rowEntries[i]

		// 对当前行的特征按列索引排序
		sort.Slice(entries, func(m, n int) bool {
			return entries[m].col < entries[n].col
		})
		// 添加特征
		for _, e := range entries {
			featureIndex := e.col
			if !zeroBase {
				featureIndex++
			}
			// 对于值为1的特征，直接输出 index:1
			if e.value == 1.0 {
				sb.WriteString(fmt.Sprintf(" %d:1", featureIndex))
			} else {
				sb.WriteString(fmt.Sprintf(" %d:%."+fmt.Sprintf("%d", precision)+"f",
					featureIndex, e.value))
			}
		}

		if i < sm.Rows-1 {
			sb.WriteString("\n")
		}
	}

	return sb.String(), nil
}

// DenseToSparse 将密集矩阵转换为稀疏矩阵
// 只保存非零元素
func DenseToSparse(dense [][]float64) *SparseMatrix {
	var data []float64
	var rowIdx []int
	var colIdx []int

	rows := len(dense)
	cols := 0
	if rows > 0 {
		cols = len(dense[0])
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if dense[i][j] != 0 {
				data = append(data, dense[i][j])
				rowIdx = append(rowIdx, i)
				colIdx = append(colIdx, j)
			}
		}
	}

	return &SparseMatrix{
		Rows:   rows,
		Cols:   cols,
		Data:   data,
		RowIdx: rowIdx,
		ColIdx: colIdx,
	}
}

func MergeMultipleSparseMatrixCols(matrices ...*SparseMatrix) (*SparseMatrix, error) {
	if len(matrices) == 0 {
		return nil, fmt.Errorf("没有提供矩阵")
	}

	// 验证所有矩阵的行数是否匹配
	numRows := matrices[0].Rows
	for _, matrix := range matrices {
		if matrix.Rows != numRows {
			return nil, fmt.Errorf("行数不匹配: %d != %d", matrix.Rows, numRows)
		}
	}

	// 计算合并后的列数和数据容量
	totalCols := 0
	totalData := 0
	for _, matrix := range matrices {
		totalCols += matrix.Cols
		totalData += len(matrix.Data)
	}

	// 创建新的稀疏矩阵
	merged := &SparseMatrix{
		Rows:   numRows,
		Cols:   totalCols,
		Data:   make([]float64, 0, totalData),
		RowIdx: make([]int, 0, totalData),
		ColIdx: make([]int, 0, totalData),
	}

	// 复制所有矩阵的数据，并调整列索引
	currentColOffset := 0
	for _, matrix := range matrices {
		merged.Data = append(merged.Data, matrix.Data...)
		for i := range matrix.ColIdx {
			merged.RowIdx = append(merged.RowIdx, matrix.RowIdx[i])
			merged.ColIdx = append(merged.ColIdx, matrix.ColIdx[i]+currentColOffset)
		}
		currentColOffset += matrix.Cols
	}
	return merged, nil
}

func MergeMultipleSparseMatrixRows(matrices ...*SparseMatrix) (*SparseMatrix, error) {
	if len(matrices) == 0 {
		return nil, fmt.Errorf("没有提供矩阵")
	}

	// 验证所有矩阵的特征维度是否匹配
	numFeatures := matrices[0].Cols
	for _, matrix := range matrices {
		if matrix.Cols != numFeatures {
			return nil, fmt.Errorf("特征维度不匹配: %d != %d", matrix.Cols, numFeatures)
		}
	}

	// 计算合并后的行数和数据容量
	totalRows := 0
	totalData := 0
	for _, matrix := range matrices {
		totalRows += matrix.Rows
		totalData += len(matrix.Data)
	}

	// 创建新的稀疏矩阵
	merged := &SparseMatrix{
		Rows:   totalRows,
		Cols:   numFeatures,
		Data:   make([]float64, 0, totalData),
		RowIdx: make([]int, 0, totalData),
		ColIdx: make([]int, 0, totalData),
	}

	// 复制所有矩阵的数据，并调整行索引
	currentRowOffset := 0
	for _, matrix := range matrices {
		merged.Data = append(merged.Data, matrix.Data...)
		for i := range matrix.RowIdx {
			merged.RowIdx = append(merged.RowIdx, matrix.RowIdx[i]+currentRowOffset)
			merged.ColIdx = append(merged.ColIdx, matrix.ColIdx[i])
		}
		currentRowOffset += matrix.Rows
	}

	return merged, nil
}
