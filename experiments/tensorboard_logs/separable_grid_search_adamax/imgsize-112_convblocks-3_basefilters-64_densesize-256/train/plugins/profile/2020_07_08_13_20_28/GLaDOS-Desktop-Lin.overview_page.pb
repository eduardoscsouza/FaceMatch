�	i��Q՛Z@i��Q՛Z@!i��Q՛Z@	~m�f��?~m�f��?!~m�f��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-i��Q՛Z@�W���?1��`�Y@I؁sF�v�?Y(-\VaS @*	�x�&�g�@2]
&Iterator::Model::FlatMap[0]::Generator�N�P�@!�ˀ#X�X@)�N�P�@1�ˀ#X�X@:Preprocessing2F
Iterator::Model�}�u��@!      Y@)�p=
ף�?1Y�^p�V�?:Preprocessing2O
Iterator::Model::FlatMapW@��>�@!����X@),+MJA�g?17�֡�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�W���?�W���?!�W���?      ��!       "	��`�Y@��`�Y@!��`�Y@*      ��!       2      ��!       :	؁sF�v�?؁sF�v�?!؁sF�v�?B      ��!       J	(-\VaS @(-\VaS @!(-\VaS @R      ��!       Z	(-\VaS @(-\VaS @!(-\VaS @JGPU�
"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown#��!���?!#��!���?"�
{gradient_tape/model/block-1_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown6n�Sm�?!���p)�?"�
{gradient_tape/model/block-0_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown9OT^C�?!�jo���?"�
{gradient_tape/model/block-1_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknownrc�t�I�?!Mj����?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownS�@�.g�?!�p#t/��?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknown�!�M���?!(�ZI�?"T
+Adamax/Adamax/update_18/ResourceApplyAdaMaxResourceApplyAdaMax�C��b�?!8jiĦj�?"z
]gradient_tape/model/block-2_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown]I
�(�?!]���H��?"z
]gradient_tape/model/block-1_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown1�ۄ?!!����?"]
@gradient_tape/model/block-0_pool/MaxPool/MaxPoolGrad:MaxPoolGradUnknown<x�?!�L�Y�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 