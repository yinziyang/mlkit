package ngram

import (
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

var (
	// tokenPool 用于复用临时的字节切片，减少内存分配
	// 主要用于存储单个token的字节数据
	// 初始容量为64字节，适用于大多数场景
	tokenPool = sync.Pool{
		New: func() interface{} {
			b := make([]byte, 0, 64)
			return &b
		},
	}

	// tokensPool 用于存储所有token的字节切片
	// 容量较大(256字节)，用于存储完整的处理文本
	tokensPool = sync.Pool{
		New: func() interface{} {
			b := make([]byte, 0, 256)
			return &b
		},
	}

	// tokenEndsPool 用于存储每个token的结束位置
	// 用于快速定位和切分token
	tokenEndsPool = sync.Pool{
		New: func() interface{} {
			s := make([]int, 0, 64)
			return &s
		},
	}

	// builderPool 用于复用strings.Builder对象
	// 用于高效的字符串拼接操作
	builderPool = sync.Pool{
		New: func() interface{} {
			return new(strings.Builder)
		},
	}
)

func NGram(text string, n uint) []string {
	// 去除首尾空格
	text = strings.TrimSpace(text)
	if n == 0 || text == "" {
		return nil
	}

	// 将字符串转换为 rune 切片以便遍历
	runeText := []rune(text)
	length := uint(len(runeText))

	// 如果 n 大于文本长度，直接返回整个文本
	if n > length {
		return []string{string(runeText)}
	}

	// 检查是否包含CJK文字或中文标点
	if !containsCJKOrPunct(text) {
		// 对于纯英文文本，使用空格分词
		return NGramSpace(text, n)
	}

	// 预分配结果切片
	result := make([]string, 0, length-n+1)

	// 从池中获取临时对象
	tokenBufPtr := tokenPool.Get().(*[]byte)
	tokensPtr := tokensPool.Get().(*[]byte)
	tokenEndsPtr := tokenEndsPool.Get().(*[]int)
	tokenBuf := (*tokenBufPtr)[:0]
	tokens := (*tokensPtr)[:0]
	tokenEnds := (*tokenEndsPtr)[:0]

	// 确保函数返回时将对象放回池中
	defer func() {
		tokenPool.Put(tokenBufPtr)
		tokensPool.Put(tokensPtr)
		tokenEndsPool.Put(tokenEndsPtr)
	}()

	// currentType 用于标记当前处理的字符类型
	// 0: 其他字符(如中文)
	// 1: 英文字母
	// 2: 数字
	currentType := 0
	lastEnd := 0 // 记录上一个token的结束位置
	// segmentStarts 存储文本分段的起始位置
	// 主要用于处理标点符号分隔的文本段
	segmentStarts := make([]int, 1, 8)

	// flushToken 将当前累积的token写入到tokens切片中
	// 同时更新lastEnd和tokenEnds
	flushToken := func() {
		if len(tokenBuf) > 0 {
			tokens = append(tokens, tokenBuf...)
			lastEnd = len(tokens)
			tokenEnds = append(tokenEnds, lastEnd)
			tokenBuf = tokenBuf[:0]
		}
	}

	// 处理字符
	for i := uint(0); i < length; i++ {
		char := runeText[i]

		// 处理分隔符
		if unicode.IsPunct(char) || unicode.IsControl(char) || unicode.IsSymbol(char) || char == ' ' {
			flushToken()
			currentType = 0
			// 标点符号只作为分隔符，不作为token
			segmentStarts = append(segmentStarts, len(tokenEnds))
			continue
		}

		if unicode.IsLetter(char) && char <= unicode.MaxASCII {
			if currentType != 1 {
				flushToken()
				currentType = 1
			}
			tokenBuf = append(tokenBuf, byte(char))
		} else if unicode.IsDigit(char) {
			if currentType != 2 {
				flushToken()
				currentType = 2
			}
			tokenBuf = append(tokenBuf, byte(char))
		} else {
			if currentType != 0 {
				flushToken()
				currentType = 0
			}
			// 对于非ASCII字符，需要转换为UTF-8编码
			if char <= unicode.MaxASCII {
				tokens = append(tokens, byte(char))
			} else {
				b := make([]byte, 4)
				n := utf8.EncodeRune(b, char)
				tokens = append(tokens, b[:n]...)
			}
			lastEnd = len(tokens)
			tokenEnds = append(tokenEnds, lastEnd)
		}
	}

	// 处理最后的token
	flushToken()
	segmentStarts = append(segmentStarts, len(tokenEnds))

	// 生成n-gram
	tokenCount := len(tokenEnds)
	if uint(tokenCount) >= n {
		// 从池中获取Builder
		sb := builderPool.Get().(*strings.Builder)
		sb.Reset()
		sb.Grow(int(n) * 4)
		defer builderPool.Put(sb)

		// 为每个分段生成n-gram
		for i := 0; i < len(segmentStarts)-1; i++ {
			segStart := segmentStarts[i]
			segEnd := segmentStarts[i+1]
			segLen := segEnd - segStart

			if uint(segLen) >= n {
				start := 0
				if i > 0 && segStart > 0 {
					start = tokenEnds[segStart-1]
				}

				for j := segStart; j <= segEnd-int(n); j++ {
					sb.Reset()
					pos := start
					for k := 0; k < int(n); k++ {
						end := tokenEnds[j+k]
						sb.Write(tokens[pos:end])
						pos = end
					}
					// 检查生成的token是否为单个rune
					token := sb.String()
					if len([]rune(token)) > 1 {
						result = append(result, token)
					}
					if j < segEnd-1 {
						start = tokenEnds[j]
					}
				}
			} else if segLen > 0 {
				// 如果分段长度小于n但大于0，将整个分段作为一个token
				sb.Reset()
				start := 0
				if i > 0 && segStart > 0 {
					start = tokenEnds[segStart-1]
				}
				end := tokenEnds[segEnd-1]
				sb.Write(tokens[start:end])
				// 检查生成的token是否为单个rune
				token := sb.String()
				runeToken := []rune(token)
				l := len(runeToken)
				if l == 1 {
					if !isDigitSymbolOrLetter(runeToken[0]) {
						result = append(result, token)
					}
				} else if l > 1 {
					if !containsOnlyControlPunctOrSymbol(runeToken) {
						result = append(result, token)
					}
				}
			}
		}
	}

	return result
}

func isDigitSymbolOrLetter(r rune) bool {
	return unicode.IsDigit(r) || unicode.IsLetter(r) || unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// isControlPunctOrSymbol 判断 rune 是否是控制符、标点符号或符号
func isControlPunctOrSymbol(r rune) bool {
	return unicode.IsControl(r) || unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// containsOnlyControlPunctOrSymbol 判断 []rune 是否只包含控制符、标点符号和符号
func containsOnlyControlPunctOrSymbol(tokens []rune) bool {
	for _, r := range tokens {
		if !isControlPunctOrSymbol(r) {
			return false
		}
	}
	return true
}

// containsCJKOrPunct 检查文本是否包含中日韩文字或中文标点符号
func containsCJKOrPunct(text string) bool {
	for _, r := range text {
		if unicode.Is(unicode.Han, r) || // 中文
			unicode.Is(unicode.Hiragana, r) || // 日文平假名
			unicode.Is(unicode.Katakana, r) || // 日文片假名
			unicode.Is(unicode.Hangul, r) {
			return true
		}
	}
	return false
}

// NGramSpace 是基于空格分词的N-gram实现
// 特点:
//   - 使用空格作为分词依据
//   - 支持标点符号分段处理
//   - 优化的内存使用
//
// 参数:
//   - text: 输入文本
//   - n: N-gram的大小
//
// 返回:
//   - []string: 处理后的N-gram切片
func NGramSpace(text string, n uint) []string {
	// 去除首尾空格
	text = strings.TrimSpace(text)
	if n == 0 || text == "" {
		return nil
	}

	// 预分配结果切片
	result := make([]string, 0, 8)

	// 按标点符号分割文本
	segments := strings.FieldsFunc(text, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsControl(r) || unicode.IsSymbol(r)
	})

	// 处理每个分段
	for i, segment := range segments {
		words := strings.Fields(segment)
		if len(words) == 0 {
			continue
		}

		// 如果是最后一个分段且词数小于n，返回整个分段
		if i == len(segments)-1 && uint(len(words)) < n {
			result = append(result, strings.Join(words, " "))
			continue
		}

		// 如果分段只有一个词，直接添加
		if len(words) == 1 {
			result = append(result, words[0])
			continue
		}

		// 如果分段词数小于n，添加每个单词
		if uint(len(words)) < n {
			result = append(result, words...)
			continue
		}

		// 生成该分段的n-gram
		for i := uint(0); i <= uint(len(words))-n; i++ {
			var sb strings.Builder
			for j := uint(0); j < n; j++ {
				if j > 0 {
					sb.WriteString(" ")
				}
				sb.WriteString(words[i+j])
			}
			result = append(result, sb.String())
		}
	}

	return result
}
