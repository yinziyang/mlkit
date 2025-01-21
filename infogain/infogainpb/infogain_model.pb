syntax = "proto3";

package infogain_model;
option go_package = "./;infogainpb";

// InfoGainModel 存储信息增益模型的相关参数
message InfoGainModel {
    int32 max_features = 1;
    map<string, double> scores = 2;   // scores 的 key 就是特征词
    int32 num_features = 3;
}