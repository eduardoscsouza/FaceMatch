�	P4`��o@P4`��o@!P4`��o@	qu3�"�@qu3�"�@!qu3�"�@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-P4`��o@�;F��?1�7k�>hn@I�[��bo�?Y�n/i��"@*	+��N;�@2]
&Iterator::Model::FlatMap[0]::Generatorc_��`�#@!|��CE�X@)c_��`�#@1|��CE�X@:Preprocessing2F
Iterator::Modeli� �w�#@!      Y@)^gE�D?1�����س?:Preprocessing2O
Iterator::Model::FlatMap��w��#@!A��	�X@)��+�pq?1�'���#�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�;F��?�;F��?!�;F��?      ��!       "	�7k�>hn@�7k�>hn@!�7k�>hn@*      ��!       2      ��!       :	�[��bo�?�[��bo�?!�[��bo�?B      ��!       J	�n/i��"@�n/i��"@!�n/i��"@R      ��!       Z	�n/i��"@�n/i��"@!�n/i��"@JGPU�
"�
{gradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown��)#��?!��)#��?"�
{gradient_tape/model/block-1_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown��$�?��?!�ʬ�a��?"�
{gradient_tape/model/block-0_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknownϷ!2-�?!�8����?"�
{gradient_tape/model/block-1_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown��]��[�?!b�} �?"�
{gradient_tape/model/block-2_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown!�>r�s�?!4��4`f�?"�
{gradient_tape/model/block-2_conv_0/separable_conv2d/DepthwiseConv2dNativeBackpropFilter:DepthwiseConv2dNativeBackpropFilterUnknown��\��?!__ ���?"z
]gradient_tape/model/block-0_conv_1/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownz��M$�?!�׾,���?"�
ygradient_tape/model/block-0_conv_1/separable_conv2d/DepthwiseConv2dNativeBackpropInput:DepthwiseConv2dNativeBackpropInputUnknown�����|z?!�Cԓ�/�?"-
IteratorGetNext/_1_Sendx��h+z?!����c�?"z
]gradient_tape/model/block-0_conv_0/separable_conv2d/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�UTGv?!*2N����?2blackQ      Y@"�
device�Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 