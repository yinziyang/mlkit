package utils

import "fmt"

type FeatureScore struct {
	Feature string
	Score   float64
}

func (fs FeatureScore) String() string {
	return fmt.Sprintf("%s: %f", fs.Feature, fs.Score)
}
