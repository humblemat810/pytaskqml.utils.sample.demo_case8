# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yolo_detection.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14yolo_detection.proto\"T\n\x0bRequestData\x12\x13\n\x0bmessageuuid\x18\x01 \x01(\t\x12\x11\n\x05image\x18\x02 \x03(\x02\x42\x02\x10\x01\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\"H\n\x03\x62\x62\x63\x12\x10\n\x08x_center\x18\x01 \x01(\x02\x12\x10\n\x08y_center\x18\x02 \x01(\x02\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\"F\n\tDetection\x12\x12\n\nclasslabel\x18\x01 \x01(\t\x12\x11\n\x03\x62\x62\x63\x18\x02 \x01(\x0b\x32\x04.bbc\x12\x12\n\nconfidence\x18\x03 \x01(\x02\"B\n\x0cResponseData\x12\x13\n\x0bmessageuuid\x18\x01 \x01(\t\x12\x1d\n\tDetection\x18\x02 \x03(\x0b\x32\n.Detectionb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'yolo_detection_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REQUESTDATA'].fields_by_name['image']._options = None
  _globals['_REQUESTDATA'].fields_by_name['image']._serialized_options = b'\020\001'
  _globals['_REQUESTDATA']._serialized_start=24
  _globals['_REQUESTDATA']._serialized_end=108
  _globals['_BBC']._serialized_start=110
  _globals['_BBC']._serialized_end=182
  _globals['_DETECTION']._serialized_start=184
  _globals['_DETECTION']._serialized_end=254
  _globals['_RESPONSEDATA']._serialized_start=256
  _globals['_RESPONSEDATA']._serialized_end=322
# @@protoc_insertion_point(module_scope)
