syntax = "proto3";


package label_encoder_model;

option go_package = "./;label_encoderpb";

message LabelEncoderModel {
  repeated string classes = 1;
  map<string, int32> label_to_index = 2;
}