�	���5��U@���5��U@!���5��U@	��id~��?��id~��?!��id~��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���5��U@��U�P��?16�o��jU@I��m3��?Y/�ͮ�?*	9��v��@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator`<��bD@!{p=���X@)`<��bD@1{p=���X@:Preprocessing2F
Iterator::Model�X�Х?!qуȣ��?)C�5v��?1s����R�?:Preprocessing2P
Iterator::Model::Prefetch�ypw�n�?!��`��͠?)�ypw�n�?1��`��͠?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap��v� cD@!��Q�X@)���n?1�:��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��U�P��?��U�P��?!��U�P��?      ��!       "	6�o��jU@6�o��jU@!6�o��jU@*      ��!       2      ��!       :	��m3��?��m3��?!��m3��?B      ��!       J	/�ͮ�?/�ͮ�?!/�ͮ�?R      ��!       Z	/�ͮ�?/�ͮ�?!/�ͮ�?JGPU�
"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�]�9Z�?!�]�9Z�?"�
{gradient_tape/model/block-0_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�پ��{�?!B�r���?"�
{gradient_tape/model/block-1_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�N�f�?!إ��N��?"�
{gradient_tape/model/block-1_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown�����?!x����h�?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�WV�+Ѓ?!� 4�2��?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknownz�����?!`�
���?"z
]gradient_tape/model/block-1_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown���&�|?!���<�?"-
IteratorGetNext/_1_Sendj��&�y?!5�]��o�?"]
@gradient_tape/model/block-0_pool/MaxPool/MaxPoolGrad:MaxPoolGradUnknown}���/uu?!�H���?"z
]gradient_tape/model/block-0_conv_0/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownz��t?!-;sG���?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 