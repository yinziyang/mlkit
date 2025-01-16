package ngram

import (
	"reflect"
	"testing"
)

func TestNGram(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		n        uint
		expected []string
	}{
		{
			name:     "纯中文2-gram",
			text:     "成人在线",
			n:        2,
			expected: []string{"成人", "人在", "在线"},
		},
		{
			name:     "中英混合2-gram",
			text:     "成人av在线",
			n:        2,
			expected: []string{"成人", "人av", "av在", "在线"},
		},
		{
			name:     "英文开头中文结尾2-gram",
			text:     "A级XXOO ",
			n:        2,
			expected: []string{"A级", "级XXOO"},
		},
		{
			name:     "中英混合3-gram",
			text:     "成人av在线",
			n:        3,
			expected: []string{"成人av", "人av在", "av在线"},
		},
		{
			name:     "中英数字混合2-gram",
			text:     "成人av123在线",
			n:        2,
			expected: []string{"成人", "人av", "av123", "123在", "在线"},
		},
		{
			name:     "中英数字混合3-gram",
			text:     "成人av123在线",
			n:        3,
			expected: []string{"成人av", "人av123", "av123在", "123在线"},
		},
		{
			name:     "英中混合2-gram",
			text:     "Hello世界",
			n:        2,
			expected: []string{"Hello世", "世界"},
		},
		{
			name:     "中文顿号分隔2-gram",
			text:     "威尼斯人、赌博、六合彩",
			n:        2,
			expected: []string{"威尼", "尼斯", "斯人", "赌博", "六合", "合彩"},
		},
		{
			name:     "输入长度小于n的情况",
			text:     "威尼斯",
			n:        4,
			expected: []string{"威尼斯"},
		},
		{
			name:     "域名2-gram",
			text:     "www.123qq.com",
			n:        2,
			expected: []string{"www", "123qq", "com"},
		},
		{
			name:     "复杂域名2-gram",
			text:     "www.123qq111.com",
			n:        2,
			expected: []string{"www", "123qq111", "com"},
		},
		{
			name:     "email2-gram",
			text:     "shop-sg11@gmail.com",
			n:        2,
			expected: []string{"shop", "sg11", "gmail", "com"},
		},
		{
			name:     "带括号的网址3-gram",
			text:     "5分钟快三(w998.cc)",
			n:        3,
			expected: []string{"5分钟", "分钟快", "钟快三", "w998", "cc"},
		},
		{
			name:     "带emoji的2-gram",
			text:     "【💎老品牌💎】🥇",
			n:        2,
			expected: []string{"老品", "品牌"},
		},
		{
			name:     "带时间和标点的2-gram",
			text:     "时间：2025-01-13 11:35:50  作者：5分钟快三 浏览量：7",
			n:        2,
			expected: []string{"时间", "2025", "01", "13", "11", "35", "50", "作者", "5分", "分钟", "钟快", "快三", "浏览", "览量"},
		},
		{
			name:     "中英空格分隔2-gram",
			text:     "你Hello 好World",
			n:        2,
			expected: []string{"你Hello", "好World"},
		},
		{
			name:     "中英文混合2-gram",
			text:     "逸影直播APP",
			n:        2,
			expected: []string{"逸影", "影直", "直播", "播APP"},
		},
		{
			name:     "中英文混合3-gram",
			text:     "逸影直播APP",
			n:        3,
			expected: []string{"逸影直", "影直播", "直播APP"},
		},
		{
			name:     "数字中文符号混合3-gram",
			text:     "5分钟快三「中国」有限公司",
			n:        3,
			expected: []string{"5分钟", "分钟快", "钟快三", "中国", "有限公", "限公司"},
		},
		{
			name:     "数字中文符号混合3-gram",
			text:     "3377体育|电竞-官方网站",
			n:        3,
			expected: []string{"3377体育", "电竞", "官方网", "方网站"},
		},
		{
			name: "数字中文符号混合3-gram",
			text: "安卓请使用谷歌(Chrome)浏览器访问，无广告/体验流畅/速度更快，iPhone请使用手机自带Safria浏览器访问。",
			n:    3,
			expected: []string{
				"安卓请", "卓请使", "请使用", "使用谷", "用谷歌", "Chrome", "浏览器", "览器访", "器访问", "无广告", "体验流", "验流畅", "速度更", "度更快", "iPhone请使", "请使用", "使用手", "用手机", "手机自", "机自带", "自带Safria", "带Safria浏", "Safria浏览", "浏览器", "览器访", "器访问",
			},
		},
		{
			name: "中文符号混合3-gram",
			text: "* 为防止丢失本站，请立即收藏该页地址，收藏并分享给好盆友。",
			n:    3,
			expected: []string{
				"为防止", "防止丢", "止丢失", "丢失本", "失本站", "请立即", "立即收", "即收藏", "收藏该", "藏该页", "该页地", "页地址", "收藏并", "藏并分", "并分享", "分享给", "享给好", "给好盆", "好盆友",
			},
		},
		{
			name:     "中英混合空格分隔2-gram",
			text:     "你好 hello world",
			n:        2,
			expected: []string{"你好", "hello", "world"},
		},

		{
			name:     "英文2-gram",
			text:     "hello world example test",
			n:        2,
			expected: []string{"hello world", "world example", "example test"},
		},
		{
			name:     "英文3-gram",
			text:     "hello world example test",
			n:        3,
			expected: []string{"hello world example", "world example test"},
		},
		{
			name:     "数字混合2-gram",
			text:     "test 123 456 789",
			n:        2,
			expected: []string{"test 123", "123 456", "456 789"},
		},
		{
			name:     "短于n的文本",
			text:     "hello world",
			n:        3,
			expected: []string{"hello world"},
		},
		{
			name:     "带标点的文本",
			text:     "hello, world! how are you",
			n:        2,
			expected: []string{"hello", "world", "how are", "are you"},
		},
		{
			name:     "纯英文空格分词",
			text:     "hello world test",
			n:        2,
			expected: []string{"hello world", "world test"},
		},
		{
			name:     "英文2-gram",
			text:     "Free Indian Porn Videos",
			n:        2,
			expected: []string{"Free Indian", "Indian Porn", "Porn Videos"},
		},
		{
			name:     "纯英文带标点",
			text:     "hello, world! test",
			n:        2,
			expected: []string{"hello", "world", "test"},
		},
		{
			name:     "英文数字混合2-gram",
			text:     "hello world 123 test",
			n:        2,
			expected: []string{"hello world", "world 123", "123 test"},
		},
		{
			name:     "英文数字混合2-gram",
			text:     "hello world123 test",
			n:        2,
			expected: []string{"hello world123", "world123 test"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NGramOptimized(tt.text, tt.n)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("NGram() = %+v, want %+v", got, tt.expected)
			}
		})
	}
}

func TestNGramSpace(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		n        uint
		expected []string
	}{
		{
			name:     "英文2-gram",
			text:     "hello world example test",
			n:        2,
			expected: []string{"hello world", "world example", "example test"},
		},
		{
			name:     "英文3-gram",
			text:     "hello world example test",
			n:        3,
			expected: []string{"hello world example", "world example test"},
		},
		{
			name:     "数字混合2-gram",
			text:     "test 123 456 789",
			n:        2,
			expected: []string{"test 123", "123 456", "456 789"},
		},
		{
			name:     "短于n的文本",
			text:     "hello world",
			n:        3,
			expected: []string{"hello world"},
		},
		{
			name:     "带标点的文本",
			text:     "hello, world! how are you",
			n:        2,
			expected: []string{"hello", "world", "how are", "are you"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NGramSpace(tt.text, tt.n)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("NGramSpace() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestNGramOptimizedWithCJKDetection(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		n        uint
		expected []string
	}{
		{
			name:     "纯英文空格分词",
			text:     "hello world test",
			n:        2,
			expected: []string{"hello world", "world test"},
		},
		{
			name:     "纯英文带标点",
			text:     "hello, world! test",
			n:        2,
			expected: []string{"hello", "world", "test"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NGramOptimized(tt.text, tt.n)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("NGramOptimized() = %v, want %v", got, tt.expected)
			}
		})
	}
}
