�	^����6x@^����6x@!^����6x@	t~ĉ=�?t~ĉ=�?!t~ĉ=�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-^����6x@m���5��?1�Ά�� x@Iګ��>�?Y^gC��A�?*	;�O�7�@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator) ���C@!��A���X@)) ���C@1��A���X@:Preprocessing2F
Iterator::Model����B��?!�?�%:��?)�):���?1���^�?:Preprocessing2P
Iterator::Model::Prefetch�g��s��?!B�l`�˛?)�g��s��?1B�l`�˛?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�o�C@!�vqW�X@)n�8)�{l?1!�F<�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	m���5��?m���5��?!m���5��?      ��!       "	�Ά�� x@�Ά�� x@!�Ά�� x@*      ��!       2      ��!       :	ګ��>�?ګ��>�?!ګ��>�?B      ��!       J	^gC��A�?^gC��A�?!^gC��A�?R      ��!       Z	^gC��A�?^gC��A�?!^gC��A�?JGPU�
"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown���3�j�?!���3�j�?"�
{gradient_tape/model/block-0_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown����ֹ?!$�Q����?"�
{gradient_tape/model/block-1_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknowneC���?!������?"�
{gradient_tape/model/block-1_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown3���z�?!���,�b�?"�
{gradient_tape/model/block-2_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknownՇ�d%P�?!$/X|]�?"�
{gradient_tape/model/block-2_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown+�Ď�?!V�܌<�?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��<��̑?!�<y4���?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknownQTu��?!7�N:	;�?"Q
4gradient_tape/model/block-0_conv_1/ReluGrad:ReluGradUnknownY��Ņ?!�����?"Q
4gradient_tape/model/block-0_conv_0/ReluGrad:ReluGradUnknown/�w�\��?!�] ���?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 