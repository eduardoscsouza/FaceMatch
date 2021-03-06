�	�vi�abW@�vi�abW@!�vi�abW@	g��@g��@!g��@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�vi�abW@�\QJ��?1J�ʽ�rU@I�;O<g��?Y�\��u@*	G�zn/�@2]
&Iterator::Model::FlatMap[0]::Generator^f�(�@!����X@)^f�(�@1����X@:Preprocessing2F
Iterator::Model���9�@!      Y@)\�J��?1�U:����?:Preprocessing2O
Iterator::Model::FlatMap/��dƻ@!��0��X@)������n?1$��N�$�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\QJ��?�\QJ��?!�\QJ��?      ��!       "	J�ʽ�rU@J�ʽ�rU@!J�ʽ�rU@*      ��!       2      ��!       :	�;O<g��?�;O<g��?!�;O<g��?B      ��!       J	�\��u@�\��u@!�\��u@R      ��!       Z	�\��u@�\��u@!�\��u@JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownP�hmK�?!P�hmK�?"T
+Adamax/Adamax/update_16/ResourceApplyAdaMaxResourceApplyAdaMax���D���?!��ΏL��?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownۋWu+)�?!S==%q[�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�'��(�?!R��r@�?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknown�x��(�?!cQ� �?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknownH��+���?!��*&���?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknown�]0y���?!��m[��?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown<
�?!K��^*��?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown���W���?!�
j!��?"i
Lgradient_tape/model/block-0_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownp0\&��?!n�=�?�?2blackQ      Y@"�
both�Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 