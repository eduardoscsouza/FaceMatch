�	÷�n��F@÷�n��F@!÷�n��F@	{/פ:�$@{/פ:�$@!{/פ:�$@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-÷�n��F@�;��?1>ʈ@7C@I`:�۠��?YM�St�@*	7�A`���@2]
&Iterator::Model::FlatMap[0]::Generator�H�,|�@!AU�R!�X@)�H�,|�@1AU�R!�X@:Preprocessing2F
Iterator::Model����]�@!      Y@)ڨN�~?1�
�ޣ5�?:Preprocessing2O
Iterator::Model::FlatMapX����@!{�.e�X@)�unڌ�p?1]��l�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 10.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�;��?�;��?!�;��?      ��!       "	>ʈ@7C@>ʈ@7C@!>ʈ@7C@*      ��!       2      ��!       :	`:�۠��?`:�۠��?!`:�۠��?B      ��!       J	M�St�@M�St�@!M�St�@R      ��!       Z	M�St�@M�St�@!M�St�@JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown���n��?!���n��?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�D�3���?!�@�/���?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown���e˅�?!u�I�Va�?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown��>�/�?!m�Cm�?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknownD����?!������?"i
Lgradient_tape/model/block-0_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�'(��e�?!�9���?"Q
4gradient_tape/model/block-1_conv_0/ReluGrad:ReluGradUnknown�Fvk-�?!�ڧ�A��?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknown���?!�|��=�?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�������?!wx�P
�?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown8�����?!L�,���?2blackQ      Y@"�
both�Your program is MODERATELY input-bound because 10.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 