package label_encoder

import (
	"fmt"
	"os"
	"sort"

	"github.com/yinziyang/mlkit/label_encoder/label_encoderpb"
	"google.golang.org/protobuf/proto"
)

// LabelEncoder 将分类标签编码为数字
type LabelEncoder struct {
	// classes 存储已知的唯一标签列表（按顺序）
	classes []string
	// labelToIndex 存储标签到索引的映射
	labelToIndex map[string]int
	// 是否已经训练过
	fitted bool
	// 类别数量
	classesNum int
}

// New 创建一个新的 LabelEncoder 实例
func New() *LabelEncoder {
	return &LabelEncoder{
		labelToIndex: make(map[string]int),
	}
}

// Fit 训练编码器
func (le *LabelEncoder) Fit(labels []string) error {
	// 创建唯一标签集合
	uniqueLabels := make(map[string]struct{})
	for _, label := range labels {
		uniqueLabels[label] = struct{}{}
	}

	// 将唯一标签转换为切片并排序
	le.classes = make([]string, 0, len(uniqueLabels))
	for label := range uniqueLabels {
		le.classes = append(le.classes, label)
		le.classesNum += 1
	}
	sort.Strings(le.classes)

	// 创建标签到索引的映射
	le.labelToIndex = make(map[string]int)
	for i, class := range le.classes {
		le.labelToIndex[class] = i
	}

	le.fitted = true
	return nil
}

// Transform 将标签转换为数字索引
func (le *LabelEncoder) Transform(labels []string) ([]int, error) {
	if !le.fitted {
		return nil, fmt.Errorf("label encoder 尚未训练，请先调用 Fit 方法")
	}

	result := make([]int, len(labels))
	for i, label := range labels {
		index, exists := le.labelToIndex[label]
		if !exists {
			return nil, fmt.Errorf("未知标签: %s", label)
		}
		result[i] = index
	}

	return result, nil
}

// FitTransform 组合了 Fit 和 Transform 的功能
func (le *LabelEncoder) FitTransform(labels []string) ([]int, error) {
	if err := le.Fit(labels); err != nil {
		return nil, err
	}
	return le.Transform(labels)
}

// GetClasses 返回已知的标签数量
func (le *LabelEncoder) GetClassesNum() int {
	return le.classesNum
}

// GetClasses 返回已知的标签列表
func (le *LabelEncoder) GetClasses() []string {
	return le.classes
}

// InverseTransform 将数字索引转换回原始标签
func (le *LabelEncoder) InverseTransform(indices []int) ([]string, error) {
	if !le.fitted {
		return nil, fmt.Errorf("label encoder 尚未训练，请先调用 Fit 方法")
	}

	result := make([]string, len(indices))
	for i, idx := range indices {
		if idx < 0 || idx >= len(le.classes) {
			return nil, fmt.Errorf("索引超出范围: %d", idx)
		}
		result[i] = le.classes[idx]
	}

	return result, nil
}

// Save 将模型保存到文件
func (le *LabelEncoder) Save(filename string) error {
	if !le.fitted {
		return fmt.Errorf("label encoder 尚未训练，无法保存")
	}

	// 构建protobuf消息
	labelToIndex := make(map[string]int32)
	for k, v := range le.labelToIndex {
		labelToIndex[k] = int32(v)
	}

	model := &label_encoderpb.LabelEncoderModel{
		Classes:      le.classes,
		LabelToIndex: labelToIndex,
	}

	// 序列化
	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("序列化模型失败: %v", err)
	}

	// 写入文件
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("写入文件失败: %v", err)
	}

	return nil
}

// Load 从文件加载模型
func (le *LabelEncoder) Load(filename string) error {
	// 读取文件
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("读取文件失败: %v", err)
	}

	// 反序列化
	model := &label_encoderpb.LabelEncoderModel{}
	if err := proto.Unmarshal(data, model); err != nil {
		return fmt.Errorf("反序列化模型失败: %v", err)
	}

	// 恢复模型状态
	le.classes = model.Classes
	le.labelToIndex = make(map[string]int)
	for k, v := range model.LabelToIndex {
		le.labelToIndex[k] = int(v)
		le.classesNum += 1
	}
	le.fitted = true

	return nil
}
