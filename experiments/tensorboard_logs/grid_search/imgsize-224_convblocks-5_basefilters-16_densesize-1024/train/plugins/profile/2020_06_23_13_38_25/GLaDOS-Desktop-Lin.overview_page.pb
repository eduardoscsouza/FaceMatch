�	~b�tE@~b�tE@!~b�tE@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$~b�tE@�ǵ�b�?1x��e�D@I��E����?*	�C�lc\�@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatorę_��C@!�;��E�X@)ę_��C@1�;��E�X@:Preprocessing2F
Iterator::Model�m�s�?!k��6��?)��qќ?1�Wm�+�?:Preprocessing2P
Iterator::Model::PrefetchS�r/0+�?!%�=�n�?)S�r/0+�?1%�=�n�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�Sȕz�C@!`�@2��X@)Lo.2n?1���3�	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ǵ�b�?�ǵ�b�?!�ǵ�b�?      ��!       "	x��e�D@x��e�D@!x��e�D@*      ��!       2      ��!       :	��E����?��E����?!��E����?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��`��?!��`��?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�[ǯ@C�?!j����?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown���%�1�?!�X?:��?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknownB�}�?!T�|/�?"i
Lgradient_tape/model/block-0_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�)��@�?!n
Bf���?"-
IteratorGetNext/_2_Recv	��x�?!��Ă�O�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownzB�~�?!#+�_�?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��]�rޚ?!���Um�?"L
%Adam/Adam/update_20/ResourceApplyAdamResourceApplyAdamݤ�S.�?!Z�:���?"]
@gradient_tape/model/block-0_pool/MaxPool/MaxPoolGrad:MaxPoolGradUnknown��{h��?!�G��f��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 