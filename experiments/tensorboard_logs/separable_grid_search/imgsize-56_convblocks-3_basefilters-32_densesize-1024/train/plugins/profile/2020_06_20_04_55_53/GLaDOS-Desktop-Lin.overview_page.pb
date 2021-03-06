�	m�i�*d3@m�i�*d3@!m�i�*d3@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$m�i�*d3@� ��%s�?1��M~��1@I?��H���?*	33333CA@2F
Iterator::Model����B��?!      Y@)u�Rz���?1Ns*�cQ@:Preprocessing2P
Iterator::Model::PrefetchEb����?!�2VVp>@)Eb����?1�2VVp>@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?8.9 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� ��%s�?� ��%s�?!� ��%s�?      ��!       "	��M~��1@��M~��1@!��M~��1@*      ��!       2      ��!       :	?��H���??��H���?!?��H���?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�	"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�e���?!�e���?"�
{gradient_tape/model/block-0_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�T���1�?!�z�uSe�?"L
%Adam/Adam/update_18/ResourceApplyAdamResourceApplyAdam��S��?!.�P�,�?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownD����ى?!s�e��?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknown�:G��?!��8C���?"z
]gradient_tape/model/block-1_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown,ge��?!���9�?"b
Emodel/block-0_conv_1/separable_conv2d/depthwise:DepthwiseConv2dNativeUnknown��if�}?!W���s�?"z
]gradient_tape/model/block-0_conv_0/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown[N+f�{?!B�	O���?"z
]gradient_tape/model/block-1_conv_0/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownn��=�vy?!s��#���?"F
)gradient_tape/model/dense-0/MatMul:MatMulUnknown�䓢�?y?!=��r�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?8.9 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 