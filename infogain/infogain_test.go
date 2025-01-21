package infogain

import (
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

// TestInfoGainFit 测试使用原始文本的Fit方法
// 验证:
// 1. 特征数量是否正确
// 2. 每个特征的信息增益分数是否正确
func TestInfoGainFit(t *testing.T) {
	texts := []string{
		"python java 编程 代码",
		"代码 开发 python 程序 测试",
		"编程 开发 测试",
		"数据 分析 python 统计",
		"机器学习 数据 分析 模型",
		"网络 服务器 安全",
		"服务器 网络 运维 监控",
	}
	targets := []string{"0", "0", "0", "1", "1", "2", "2"}

	expectedScores := map[string]float64{
		"分析":     0.8631,
		"数据":     0.8631,
		"服务器":    0.8631,
		"网络":     0.8631,
		"代码":     0.4696,
		"开发":     0.4696,
		"测试":     0.4696,
		"编程":     0.4696,
		"机器学习":   0.3060,
		"模型":     0.3060,
		"统计":     0.3060,
		"python": 0.3060,
		"安全":     0.3060,
		"监控":     0.3060,
		"运维":     0.3060,
		"java":   0.1981,
		"程序":     0.1981,
	}

	ig := NewInfoGain()
	ig.Fit(texts, targets, strings.Fields)

	// 检查特征数量
	if len(ig.features) != len(expectedScores) {
		t.Errorf("特征数量不匹配: 期望 %d, 得到 %d", len(expectedScores), len(ig.features))
	}

	// 检查每个特征的信息增益分数
	tolerance := 0.0001
	for feature, expectedScore := range expectedScores {
		score, exists := ig.scores[feature]
		if !exists {
			t.Errorf("特征 '%s' 未找到", feature)
			continue
		}
		if math.Abs(score-expectedScore) > tolerance {
			t.Errorf("特征 '%s' 的信息增益分数不匹配: 期望 %.4f, 得到 %.4f", feature, expectedScore, score)
		}
	}
}

// TestInfoGainFitWithTokens 测试使用已分词文本的FitWithTokens方法
// 验证:
// 1. 特征数量是否正确
// 2. 每个特征的信息增益分数是否正确
func TestInfoGainFitWithTokens(t *testing.T) {
	tokens := [][]string{
		{"python", "java", "编程", "代码"},
		{"代码", "开发", "python", "程序", "测试"},
		{"编程", "开发", "测试"},
		{"数据", "分析", "python", "统计"},
		{"机器学习", "数据", "分析", "模型"},
		{"网络", "服务器", "安全"},
		{"服务器", "网络", "运维", "监控"},
	}
	targets := []string{"0", "0", "0", "1", "1", "2", "2"}

	expectedScores := map[string]float64{
		"分析":     0.8631,
		"数据":     0.8631,
		"服务器":    0.8631,
		"网络":     0.8631,
		"代码":     0.4696,
		"开发":     0.4696,
		"测试":     0.4696,
		"编程":     0.4696,
		"机器学习":   0.3060,
		"模型":     0.3060,
		"统计":     0.3060,
		"python": 0.3060,
		"安全":     0.3060,
		"监控":     0.3060,
		"运维":     0.3060,
		"java":   0.1981,
		"程序":     0.1981,
	}

	ig := NewInfoGain()
	ig.FitWithTokens(tokens, targets)

	// 检查特征数量
	if len(ig.features) != len(expectedScores) {
		t.Errorf("特征数量不匹配: 期望 %d, 得到 %d", len(expectedScores), len(ig.features))
	}

	// 检查每个特征的信息增益分数
	tolerance := 0.0001
	for feature, expectedScore := range expectedScores {
		score, exists := ig.scores[feature]
		if !exists {
			t.Errorf("特征 '%s' 未找到", feature)
			continue
		}
		if math.Abs(score-expectedScore) > tolerance {
			t.Errorf("特征 '%s' 的信息增益分数不匹配: 期望 %.4f, 得到 %.4f", feature, expectedScore, score)
		}
	}
}

// TestInfoGainFitWithTokensMaxFeatures 测试maxFeatures参数的功能
// 验证当maxFeatures=2时:
// 1. 每个类别是否选择了2个最高信息增益的特征
// 2. 特征的信息增益分数是否正确
// 3. 特征列表是否按字母顺序排序
// 4. 词汇表是否正确更新
func TestInfoGainFitWithTokensMaxFeatures(t *testing.T) {
	tokens := [][]string{
		{"python", "java", "编程", "代码"},
		{"代码", "开发", "python", "程序", "测试"},
		{"编程", "开发", "测试"},
		{"数据", "分析", "python", "统计"},
		{"机器学习", "数据", "分析", "模型"},
		{"网络", "服务器", "安全"},
		{"服务器", "网络", "运维", "监控"},
	}
	targets := []string{"0", "0", "0", "1", "1", "2", "2"}

	// 每个类别选择信息增益最高的两个特征
	expectedScores := map[string]float64{
		"代码":  0.4696,
		"开发":  0.4696,
		"分析":  0.8631,
		"数据":  0.8631,
		"服务器": 0.8631,
		"网络":  0.8631,
	}

	ig := NewInfoGain(2) // 每个类别选择2个特征
	ig.FitWithTokens(tokens, targets)

	// 检查特征数量
	if len(ig.features) != len(expectedScores) {
		t.Errorf("特征数量不匹配: 期望 %d, 得到 %d", len(expectedScores), len(ig.features))
	}

	// 检查每个特征的信息增益分数
	tolerance := 0.0001
	for feature, expectedScore := range expectedScores {
		score, exists := ig.scores[feature]
		if !exists {
			t.Errorf("特征 '%s' 未找到", feature)
			continue
		}
		if math.Abs(score-expectedScore) > tolerance {
			t.Errorf("特征 '%s' 的信息增益分数不匹配: 期望 %.4f, 得到 %.4f", feature, expectedScore, score)
		}
	}

	// 检查特征列表
	expectedFeatures := []string{"代码", "分析", "开发", "数据", "服务器", "网络"}
	if !reflect.DeepEqual(ig.features, expectedFeatures) {
		t.Errorf("特征列表不匹配: 期望 %v, 得到 %v", expectedFeatures, ig.features)
	}

	// 检查词汇表
	if len(ig.vocab) != len(expectedFeatures) {
		t.Errorf("词汇表大小不匹配: 期望 %d, 得到 %d", len(expectedFeatures), len(ig.vocab))
	}
	for _, feature := range expectedFeatures {
		if !ig.vocab[feature] {
			t.Errorf("词汇表中缺少特征: %s", feature)
		}
	}
}

// TestInfoGainSaveLoad 测试模型的保存和加载功能
// 验证:
// 1. 模型能否正确保存到文件
// 2. 能否从文件正确加载模型
// 3. 加载后的模型参数是否与原始模型一致:
//   - maxFeatures
//   - numFeatures
//   - 特征列表
//   - 特征分数
//   - 词汇表
func TestInfoGainSaveLoad(t *testing.T) {
	// 创建测试数据
	tokens := [][]string{
		{"python", "java", "编程", "代码"},
		{"代码", "开发", "python", "程序", "测试"},
		{"编程", "开发", "测试"},
		{"数据", "分析", "python", "统计"},
		{"机器学习", "数据", "分析", "模型"},
		{"网络", "服务器", "安全"},
		{"服务器", "网络", "运维", "监控"},
	}
	targets := []string{"0", "0", "0", "1", "1", "2", "2"}

	// 训练原始模型
	originalIG := NewInfoGain(2)
	originalIG.FitWithTokens(tokens, targets)

	// 创建临时文件
	tmpfile, err := os.CreateTemp("", "infogain_test")
	if err != nil {
		t.Fatalf("无法创建临时文件: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	// 保存模型
	if err := originalIG.Save(tmpfile.Name()); err != nil {
		t.Fatalf("保存模型失败: %v", err)
	}

	// 加载模型
	loadedIG := NewInfoGain()
	if err := loadedIG.Load(tmpfile.Name()); err != nil {
		t.Fatalf("加载模型失败: %v", err)
	}

	// 比较两个模型的参数
	if originalIG.maxFeatures != loadedIG.maxFeatures {
		t.Errorf("maxFeatures不匹配: 期望 %d, 得到 %d", originalIG.maxFeatures, loadedIG.maxFeatures)
	}

	if originalIG.numFeatures != loadedIG.numFeatures {
		t.Errorf("numFeatures不匹配: 期望 %d, 得到 %d", originalIG.numFeatures, loadedIG.numFeatures)
	}

	// 比较特征列表
	if !reflect.DeepEqual(originalIG.features, loadedIG.features) {
		t.Errorf("特征列表不匹配: \n期望 %v, \n得到 %v", originalIG.features, loadedIG.features)
	}

	// 比较分数
	tolerance := 0.0001
	for feature, originalScore := range originalIG.scores {
		loadedScore, exists := loadedIG.scores[feature]
		if !exists {
			t.Errorf("加载的模型中缺少特征 '%s'", feature)
			continue
		}
		if math.Abs(originalScore-loadedScore) > tolerance {
			t.Errorf("特征 '%s' 的分数不匹配: 期望 %.4f, 得到 %.4f", feature, originalScore, loadedScore)
		}
	}

	// 比较词汇表
	if !reflect.DeepEqual(originalIG.vocab, loadedIG.vocab) {
		t.Errorf("词汇表不匹配: \n期望 %v, \n得到 %v", originalIG.vocab, loadedIG.vocab)
	}
}
