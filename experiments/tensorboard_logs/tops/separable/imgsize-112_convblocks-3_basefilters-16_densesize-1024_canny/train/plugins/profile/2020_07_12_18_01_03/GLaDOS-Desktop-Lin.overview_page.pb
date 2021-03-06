�	O��'��I@O��'��I@!O��'��I@	��/��?��/��?!��/��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-O��'��I@���6p�?16Vb���H@I"�k^�Y�?Y���Ty�?*���S�ʇ@)      @=2]
&Iterator::Model::FlatMap[0]::Generator�P29�3�?!��G���X@)�P29�3�?1��G���X@:Preprocessing2F
Iterator::Model8en�]�?!      Y@)�����l?1+��M�?:Preprocessing2O
Iterator::Model::FlatMap��n�@�?!nZ��X@)W��x��Y?1t���F�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���6p�?���6p�?!���6p�?      ��!       "	6Vb���H@6Vb���H@!6Vb���H@*      ��!       2      ��!       :	"�k^�Y�?"�k^�Y�?!"�k^�Y�?B      ��!       J	���Ty�?���Ty�?!���Ty�?R      ��!       Z	���Ty�?���Ty�?!���Ty�?JGPU�
"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknownt�����?!t�����?"�
{gradient_tape/model/block-1_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�:��S �?!���H�q�?"�
{gradient_tape/model/block-1_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown���JL �?!�V��1�?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknown@ӹ8�ȡ?!��uiN�?"L
%Adam/Adam/update_18/ResourceApplyAdamResourceApplyAdam����6�?!�h��?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�&]U/��?!�k�ҢN�?"Q
4gradient_tape/model/block-2_conv_0/ReluGrad:ReluGradUnknown}F�B�i�?!���+I��?"z
]gradient_tape/model/block-0_conv_0/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��l�-|?!�y�����?"x
[gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropInput:Conv2DBackpropInputUnknowne���v?!<�ئ���?"z
]gradient_tape/model/block-1_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown^��Kyfs?!KVp��#�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 