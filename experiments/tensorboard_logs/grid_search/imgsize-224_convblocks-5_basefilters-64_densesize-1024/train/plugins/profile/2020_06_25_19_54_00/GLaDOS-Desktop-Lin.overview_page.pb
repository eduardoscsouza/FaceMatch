�	WAt��j@WAt��j@!WAt��j@	�Vi���?�Vi���?!�Vi���?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-WAt��j@�WuV��?1^f�(�j@ID�M��?YV}��b�?*	�Mb���@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�Cl�pDE@!��g_��X@)�Cl�pDE@1��g_��X@:Preprocessing2F
Iterator::Model+ٱ�ץ?!Z�a����?)I0��Z
�?1/��e�ղ?:Preprocessing2P
Iterator::Model::Prefetch��r�4�?!��@��?�?)��r�4�?1��@��?�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?9
EE@!�'���X@)����r?1��q�ǅ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�WuV��?�WuV��?!�WuV��?      ��!       "	^f�(�j@^f�(�j@!^f�(�j@*      ��!       2      ��!       :	D�M��?D�M��?!D�M��?B      ��!       J	V}��b�?V}��b�?!V}��b�?R      ��!       Z	V}��b�?V}��b�?!V}��b�?JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownj��n���?!j��n���?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknownھ(z�?!���d��?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknownqs��Y�?!�����9�?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��s��Ф?!|��zn�?"i
Lgradient_tape/model/block-4_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown
kf~t�?!��2��%�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�Ty�e~�?!��!@cU�?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknownPv6�ѕ�?!e��{h�?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown�,u*�?!a�{��Z�?"?
"model/block-4_conv_1/Conv2D:Conv2DUnknown�8�	N̙?!�����?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown_^�\���?!�Z����?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 