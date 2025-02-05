// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v5.26.1
// source: label_encoder.pb

package label_encoderpb

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type LabelEncoderModel struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Classes      []string         `protobuf:"bytes,1,rep,name=classes,proto3" json:"classes,omitempty"`
	LabelToIndex map[string]int32 `protobuf:"bytes,2,rep,name=label_to_index,json=labelToIndex,proto3" json:"label_to_index,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
}

func (x *LabelEncoderModel) Reset() {
	*x = LabelEncoderModel{}
	if protoimpl.UnsafeEnabled {
		mi := &file_label_encoder_pb_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LabelEncoderModel) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LabelEncoderModel) ProtoMessage() {}

func (x *LabelEncoderModel) ProtoReflect() protoreflect.Message {
	mi := &file_label_encoder_pb_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LabelEncoderModel.ProtoReflect.Descriptor instead.
func (*LabelEncoderModel) Descriptor() ([]byte, []int) {
	return file_label_encoder_pb_rawDescGZIP(), []int{0}
}

func (x *LabelEncoderModel) GetClasses() []string {
	if x != nil {
		return x.Classes
	}
	return nil
}

func (x *LabelEncoderModel) GetLabelToIndex() map[string]int32 {
	if x != nil {
		return x.LabelToIndex
	}
	return nil
}

var File_label_encoder_pb protoreflect.FileDescriptor

var file_label_encoder_pb_rawDesc = []byte{
	0x0a, 0x10, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x72, 0x2e,
	0x70, 0x62, 0x12, 0x13, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65,
	0x72, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x22, 0xce, 0x01, 0x0a, 0x11, 0x4c, 0x61, 0x62, 0x65,
	0x6c, 0x45, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x72, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x18, 0x0a,
	0x07, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x09, 0x52, 0x07,
	0x63, 0x6c, 0x61, 0x73, 0x73, 0x65, 0x73, 0x12, 0x5e, 0x0a, 0x0e, 0x6c, 0x61, 0x62, 0x65, 0x6c,
	0x5f, 0x74, 0x6f, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32,
	0x38, 0x2e, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x5f, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x72, 0x5f,
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x2e, 0x4c, 0x61, 0x62, 0x65, 0x6c, 0x45, 0x6e, 0x63, 0x6f, 0x64,
	0x65, 0x72, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x2e, 0x4c, 0x61, 0x62, 0x65, 0x6c, 0x54, 0x6f, 0x49,
	0x6e, 0x64, 0x65, 0x78, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x0c, 0x6c, 0x61, 0x62, 0x65, 0x6c,
	0x54, 0x6f, 0x49, 0x6e, 0x64, 0x65, 0x78, 0x1a, 0x3f, 0x0a, 0x11, 0x4c, 0x61, 0x62, 0x65, 0x6c,
	0x54, 0x6f, 0x49, 0x6e, 0x64, 0x65, 0x78, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03,
	0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x14,
	0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x76,
	0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x42, 0x14, 0x5a, 0x12, 0x2e, 0x2f, 0x3b, 0x6c,
	0x61, 0x62, 0x65, 0x6c, 0x5f, 0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x72, 0x70, 0x62, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_label_encoder_pb_rawDescOnce sync.Once
	file_label_encoder_pb_rawDescData = file_label_encoder_pb_rawDesc
)

func file_label_encoder_pb_rawDescGZIP() []byte {
	file_label_encoder_pb_rawDescOnce.Do(func() {
		file_label_encoder_pb_rawDescData = protoimpl.X.CompressGZIP(file_label_encoder_pb_rawDescData)
	})
	return file_label_encoder_pb_rawDescData
}

var file_label_encoder_pb_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_label_encoder_pb_goTypes = []interface{}{
	(*LabelEncoderModel)(nil), // 0: label_encoder_model.LabelEncoderModel
	nil,                       // 1: label_encoder_model.LabelEncoderModel.LabelToIndexEntry
}
var file_label_encoder_pb_depIdxs = []int32{
	1, // 0: label_encoder_model.LabelEncoderModel.label_to_index:type_name -> label_encoder_model.LabelEncoderModel.LabelToIndexEntry
	1, // [1:1] is the sub-list for method output_type
	1, // [1:1] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_label_encoder_pb_init() }
func file_label_encoder_pb_init() {
	if File_label_encoder_pb != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_label_encoder_pb_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LabelEncoderModel); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_label_encoder_pb_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_label_encoder_pb_goTypes,
		DependencyIndexes: file_label_encoder_pb_depIdxs,
		MessageInfos:      file_label_encoder_pb_msgTypes,
	}.Build()
	File_label_encoder_pb = out.File
	file_label_encoder_pb_rawDesc = nil
	file_label_encoder_pb_goTypes = nil
	file_label_encoder_pb_depIdxs = nil
}
