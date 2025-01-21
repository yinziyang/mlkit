syntax = "proto3";

package infogain_model;
option go_package = "./;infogainpb";

// InfoGainModel 存储信息增益模型的相关参数
message InfoGainModel {
    int32 max_features = 1;
    map<string, int32> feature_to_index = 2;  // 特征到索引的映射
    map<string, double> scores = 3;   // scores 的 key 就是特征词
    int32 num_features = 4;
}