package infogain

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"

	"github.com/yinziyang/mlkit/infogain/infogainpb"
	"github.com/yinziyang/mlkit/matrix"
	"github.com/yinziyang/mlkit/utils"
	"google.golang.org/protobuf/proto"
)

// InfoGain 实现了基于信息增益的特征选择算法
// 对于每个类别，选择信息增益最高的N个特征
type InfoGain struct {
	maxFeatures    int                       // 每个类别的最大特征数，如果为0或负数则保留所有特征
	vocab          map[string]bool           // 词汇表，保存所有选中的特征
	features       []string                  // 已排序的特征列表
	scores         map[string]float64        // 特征的信息增益分数
	numFeatures    int                       // 特征总数
	featureInLabel map[string]map[string]int // 特征在每个类别中的出现次数
	targets        []string                  // 标签列表(去重)
}

// NewInfoGain 创建信息增益模型
// maxFeatures 参数说明：
// - 如果设置为正数n，则每个类别选择信息增益分数最高的前n个特征
// - 如果设置为0或负数，则保留所有特征
// - 如果不设置，则默认保留所有特征
func NewInfoGain(maxFeatures ...int) *InfoGain {
	ig := &InfoGain{
		// tokenizer: tokenizer,
		vocab:  make(map[string]bool),
		scores: make(map[string]float64),
	}

	if len(maxFeatures) > 0 {
		ig.maxFeatures = maxFeatures[0]
	}

	return ig
}

// Fit 使用文本数据和对应的标签训练模型
// texts: 输入的文本列表
// targets: 对应的标签列表
// tokenizer: 分词函数，用于将文本转换为词列表
func (ig *InfoGain) Fit(texts []string, targets []string, tokenizer func(string) []string) {
	// 先对所有文本进行分词
	tokens := make([][]string, len(texts))
	for i, text := range texts {
		tokens[i] = tokenizer(text)
	}

	// 调用FitWithTokens
	ig.FitWithTokens(tokens, targets)
}

// Transform 将文本转换为特征矩阵
// texts: 输入的文本列表
// normalize: 是否对特征值进行L2归一化
// tokenizer: 分词函数
// 返回值:
// - 稀疏矩阵表示的特征矩阵
// - 特征名列表
func (ig *InfoGain) Transform(texts []string, normalize bool, tokenizer func(string) []string) (*matrix.SparseMatrix, []string) {
	tokens := make([][]string, len(texts))
	for i, text := range texts {
		tokens[i] = tokenizer(text)
	}
	return ig.TransformWithTokens(tokens, normalize)
}

// FitTransform 组合了Fit和Transform的功能
// 先训练模型，然后将输入文本转换为特征矩阵
func (ig *InfoGain) FitTransform(texts []string, targets []string, normalize bool, tokenizer func(string) []string) (*matrix.SparseMatrix, []string) {
	ig.Fit(texts, targets, tokenizer)
	return ig.Transform(texts, normalize, tokenizer)
}

// FitWithTokens 使用已分词的文本数据训练模型
// tokens: 已分词的文本列表，每个文本是一个词列表
// targets: 对应的标签列表
func (ig *InfoGain) FitWithTokens(tokens [][]string, targets []string) {
	// 统计标签频率
	targetFreq := make(map[string]int)
	for _, target := range targets {
		if _, ok := targetFreq[target]; !ok {
			ig.targets = append(ig.targets, target)
		}
		targetFreq[target]++
	}

	// 计算标签熵
	totalDocs := float64(len(targets))
	labelEntropy := 0.0
	for _, freq := range targetFreq {
		p := float64(freq) / totalDocs
		labelEntropy -= p * math.Log2(p)
	}

	// 统计每个特征在每个标签中的频率
	ig.featureInLabel = make(map[string]map[string]int)
	featureFreq := make(map[string]int)
	var mutex sync.Mutex

	// 并发处理文档
	numWorkers := runtime.GOMAXPROCS(0)
	chunkSize := (len(tokens) + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize
		if end > len(tokens) {
			end = len(tokens)
		}

		go func(start, end int) {
			defer wg.Done()
			localFeatureInLabel := make(map[string]map[string]int)
			localFeatureFreq := make(map[string]int)

			for idx := start; idx < end; idx++ {
				target := targets[idx]
				seenFeatures := make(map[string]bool)

				for _, token := range tokens[idx] {
					if !seenFeatures[token] {
						seenFeatures[token] = true
						localFeatureFreq[token]++
						if localFeatureInLabel[token] == nil {
							localFeatureInLabel[token] = make(map[string]int)
						}
						localFeatureInLabel[token][target]++
					}
				}
			}

			// 合并局部结果到全局
			mutex.Lock()
			for feature, freq := range localFeatureFreq {
				featureFreq[feature] += freq
				if ig.featureInLabel[feature] == nil {
					ig.featureInLabel[feature] = make(map[string]int)
				}
				for target, count := range localFeatureInLabel[feature] {
					ig.featureInLabel[feature][target] += count
				}
			}
			mutex.Unlock()
		}(start, end)
	}
	wg.Wait()

	// 并发计算信息增益
	type featureScore struct {
		feature string
		score   float64
	}
	scoresChan := make(chan featureScore, len(featureFreq))
	semaphore := make(chan struct{}, numWorkers)

	for feature, freq := range featureFreq {
		semaphore <- struct{}{}
		go func(feature string, freq int) {
			defer func() { <-semaphore }()

			// 计算条件熵
			conditionalEntropy := calculateFeatureEntropy(ig.featureInLabel[feature], targetFreq, totalDocs, float64(freq))

			// 计算信息增益
			informationGain := labelEntropy - conditionalEntropy

			scoresChan <- featureScore{feature, informationGain}
		}(feature, freq)
	}

	// 收集结果
	ig.scores = make(map[string]float64)
	for i := 0; i < len(featureFreq); i++ {
		score := <-scoresChan
		ig.scores[score.feature] = score.score
	}

	// 选择特征
	var scoreSlice []utils.FeatureScore
	for feature, score := range ig.scores {
		scoreSlice = append(scoreSlice, utils.FeatureScore{Feature: feature, Score: score})
	}

	// 按分数排序
	sort.Slice(scoreSlice, func(i, j int) bool {
		if scoreSlice[i].Score == scoreSlice[j].Score {
			return scoreSlice[i].Feature < scoreSlice[j].Feature
		}
		return scoreSlice[i].Score > scoreSlice[j].Score
	})

	// 选择特征
	if ig.maxFeatures > 0 {
		ig.features = make([]string, 0, ig.maxFeatures)
	} else {
		ig.features = make([]string, 0, len(scoreSlice))
	}

	newScores := make(map[string]float64)

	var labelFeatureStat = make(map[string]int)
	for i := 0; i < len(scoreSlice); i++ {
		feature := scoreSlice[i].Feature
		if ig.maxFeatures > 0 {
			for target := range ig.featureInLabel[feature] {
				if labelFeatureStat[target] < ig.maxFeatures {
					ig.features = append(ig.features, feature)
					newScores[feature] = scoreSlice[i].Score
					labelFeatureStat[target]++
					break
				}
			}
		} else {
			ig.features = append(ig.features, feature)
			newScores[feature] = scoreSlice[i].Score
		}
	}
	// 更新词汇表
	ig.vocab = make(map[string]bool, len(ig.features))
	ig.scores = newScores

	for _, feature := range ig.features {
		ig.vocab[feature] = true
	}

	sort.Strings(ig.features)
	ig.numFeatures = len(ig.features)
}

// TransformWithTokens 将已分词的文本转换为特征矩阵
// tokens: 已分词的文本列表
// normalize: 是否对特征值进行L2归一化
// 返回值:
// - 稀疏矩阵表示的特征矩阵
// - 特征名列表
func (ig *InfoGain) TransformWithTokens(tokens [][]string, normalize bool) (*matrix.SparseMatrix, []string) {
	numWorkers := runtime.NumCPU()
	rows := len(tokens)
	cols := len(ig.features)

	type transformTask struct {
		docIdx int
		words  []string
	}

	type transformResult struct {
		docIdx     int
		nonZeros   []float64
		rowIndices []int
		colIndices []int
	}

	taskChan := make(chan transformTask, numWorkers)
	resultChan := make(chan transformResult, numWorkers)
	var wg sync.WaitGroup

	// 启动工作协程池
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range taskChan {
				words := task.words
				wordPresent := make(map[string]bool)
				for _, word := range words {
					wordPresent[word] = true
				}

				// 计算特征值
				docScores := make(map[string]float64)
				for _, feature := range ig.features {
					if wordPresent[feature] {
						docScores[feature] = ig.scores[feature]
					}
				}

				// L2归一化
				if normalize {
					var norm float64
					for _, score := range docScores {
						norm += score * score
					}
					norm = math.Sqrt(norm)
					if norm > 0 {
						for feature := range docScores {
							docScores[feature] /= norm
						}
					}
				}

				// 构建局部稀疏矩阵数据
				var nonZeros []float64
				var rowIndices []int
				var colIndices []int

				for j, feature := range ig.features {
					if score := docScores[feature]; score > 0 {
						nonZeros = append(nonZeros, score)
						rowIndices = append(rowIndices, task.docIdx)
						colIndices = append(colIndices, j)
					}
				}

				resultChan <- transformResult{
					docIdx:     task.docIdx,
					nonZeros:   nonZeros,
					rowIndices: rowIndices,
					colIndices: colIndices,
				}
			}
		}()
	}
	// 发送任务
	go func() {
		for i, words := range tokens {
			taskChan <- transformTask{i, words}
		}
		close(taskChan)
	}()

	// 等待所有转换完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	var allNonZeros []float64
	var allRowIndices []int
	var allColIndices []int

	for result := range resultChan {
		allNonZeros = append(allNonZeros, result.nonZeros...)
		allRowIndices = append(allRowIndices, result.rowIndices...)
		allColIndices = append(allColIndices, result.colIndices...)
	}

	// 修改返回值部分
	return &matrix.SparseMatrix{
		Rows:   rows,
		Cols:   cols,
		Data:   allNonZeros,
		RowIdx: allRowIndices,
		ColIdx: allColIndices,
	}, ig.features
}

// FitTransformWithTokens 组合了FitWithTokens和TransformWithTokens的功能
// 先训练模型，然后将已分词的文本转换为特征矩阵
func (ig *InfoGain) FitTransformWithTokens(tokens [][]string, targets []string, normalize bool) (*matrix.SparseMatrix, []string) {
	ig.FitWithTokens(tokens, targets)
	return ig.TransformWithTokens(tokens, normalize)
}

// calculateFeatureEntropy 计算特征的条件熵
// featureLabelFreq: 特征在每个标签中的频率
// targetFreq: 每个标签的频率
// totalDocs: 文档总数
// featureCount: 特征出现的总次数
// 返回值: 特征的条件熵
func calculateFeatureEntropy(featureLabelFreq map[string]int, targetFreq map[string]int, totalDocs float64, featureCount float64) float64 {
	// 特征出现时的条件熵
	entropyPresent := 0.0
	// 特征不出现时的条件熵
	entropyAbsent := 0.0

	// 计算特征出现时的概率 P(X=1)
	pFeaturePresent := featureCount / totalDocs

	// 对每个标签计算：
	for target, labelCount := range targetFreq {
		// 特征在该标签中出现的次数
		featureInLabelCount := float64(featureLabelFreq[target])

		// 计算特征出现时的条件概率 P(Y=y|X=1)
		if featureCount > 0 {
			p := featureInLabelCount / featureCount
			if p > 0 {
				entropyPresent -= p * math.Log2(p)
			}
		}

		// 计算特征不出现时的条件概率 P(Y=y|X=0)
		featureAbsentCount := float64(labelCount) - featureInLabelCount
		totalAbsentCount := totalDocs - featureCount
		if totalAbsentCount > 0 {
			p := featureAbsentCount / totalAbsentCount
			if p > 0 {
				entropyAbsent -= p * math.Log2(p)
			}
		}
	}

	// 计算条件熵 H(Y|X) = P(X=1)H(Y|X=1) + P(X=0)H(Y|X=0)
	return pFeaturePresent*entropyPresent + (1-pFeaturePresent)*entropyAbsent
}

// GetVocab 获取词汇表
// 返回值: 包含所有选中特征的map
func (ig *InfoGain) GetVocab() map[string]bool {
	return ig.vocab
}

// GetFeatures 返回按信息增益分数排序的特征列表及其分数
func (ig *InfoGain) GetFeatureScores() []utils.FeatureScore {
	features := make([]utils.FeatureScore, len(ig.features))
	for i, feature := range ig.features {
		features[i] = utils.FeatureScore{
			Feature: feature,
			Score:   ig.scores[feature],
		}
	}
	return features
}

// Save 将模型保存到文件
// filename: 保存的文件路径
// 返回值: 错误信息
func (ig *InfoGain) Save(filename string) error {
	model := &infogainpb.InfoGainModel{
		MaxFeatures: int32(ig.maxFeatures),
		Scores:      ig.scores,
		NumFeatures: int32(ig.numFeatures),
	}

	data, err := proto.Marshal(model)
	if err != nil {
		return fmt.Errorf("无法序列化模型: %v", err)
	}

	return os.WriteFile(filename, data, 0644)
}

// Load 从文件加载模型
// filename: 模型文件路径
// 返回值: 错误信息
func (ig *InfoGain) Load(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("无法读取文件: %v", err)
	}

	model := &infogainpb.InfoGainModel{}
	if err := proto.Unmarshal(data, model); err != nil {
		return fmt.Errorf("无法反序列化模型: %v", err)
	}

	ig.maxFeatures = int(model.GetMaxFeatures())
	ig.scores = model.GetScores()

	// 按分数重建特征列表
	var scoreSlice []utils.FeatureScore
	for word, score := range ig.scores {
		scoreSlice = append(scoreSlice, utils.FeatureScore{Feature: word, Score: score})
	}
	sort.Slice(scoreSlice, func(i, j int) bool {
		if scoreSlice[i].Score == scoreSlice[j].Score {
			return scoreSlice[i].Feature < scoreSlice[j].Feature
		}
		return scoreSlice[i].Score > scoreSlice[j].Score
	})

	// 重建词汇表和特征列表
	ig.vocab = make(map[string]bool)
	ig.features = make([]string, 0, len(ig.scores))
	for _, fs := range scoreSlice {
		ig.vocab[fs.Feature] = true
		ig.features = append(ig.features, fs.Feature)
	}

	sort.Strings(ig.features)
	ig.numFeatures = int(model.GetNumFeatures())

	if len(ig.features) != ig.numFeatures {
		panic(fmt.Sprintf("infogain %d != %d", len(ig.features), ig.numFeatures))
	}

	return nil
}
