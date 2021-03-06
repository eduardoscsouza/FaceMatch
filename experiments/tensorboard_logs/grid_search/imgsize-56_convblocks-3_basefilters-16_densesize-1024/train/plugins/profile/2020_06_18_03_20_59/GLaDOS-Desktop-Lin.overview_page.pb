�	R����?@R����?@!R����?@	'V�iU@'V�iU@!'V�iU@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-R����?@.rOWw�?1��z6K@IL�����?Y]T�<;@*	��|?���@2]
&Iterator::Model::FlatMap[0]::Generator�d�,�Y;@!*SS�"�X@)�d�,�Y;@1*SS�"�X@:Preprocessing2F
Iterator::Model||Bv�Z;@!      Y@)Ly �Hc?1�u�����?:Preprocessing2O
Iterator::Model::FlatMap�x�0DZ;@!�cs�X@)N�E� V?1���t?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 85.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?3.3 % of the total step time sampled is spent on Kernel Launch.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.rOWw�?.rOWw�?!.rOWw�?      ��!       "	��z6K@��z6K@!��z6K@*      ��!       2      ��!       :	L�����?L�����?!L�����?B      ��!       J	]T�<;@]T�<;@!]T�<;@R      ��!       Z	]T�<;@]T�<;@!]T�<;@JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�E kj�?!�E kj�?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown��w�X�?!6q|%�?"L
%Adam/Adam/update_12/ResourceApplyAdamResourceApplyAdamx/�8�?!�ܑ� �?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��κ_(�?!=H@Q��?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknownI��d�?!��lt1�?"i
Lgradient_tape/model/block-0_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�`��\�?!�(\�t�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownZ��턡?!/���=��?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknownZn��ޠ?!������?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��	����?!e������?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknown����?!S\j���?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 85.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?3.3 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 