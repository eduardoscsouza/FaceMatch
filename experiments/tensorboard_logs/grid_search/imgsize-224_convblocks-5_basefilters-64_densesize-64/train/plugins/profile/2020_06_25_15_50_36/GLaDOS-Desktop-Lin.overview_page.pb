�	�I����k@�I����k@!�I����k@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�I����k@��)U�?1�B�iޞk@I��n�@�?*	�S�/��@2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorN_��,?F@!�*[֯�X@)N_��,?F@1�*[֯�X@:Preprocessing2F
Iterator::Modely��"���?! B)]���?)ǁW˝�?1�O#¬?:Preprocessing2P
Iterator::Model::Prefetch��vۅ�?!Thf.'3�?)��vۅ�?1Thf.'3�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap5ӽN�?F@!��(���X@)5�|�ݮw?1E�`�y��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��)U�?��)U�?!��)U�?      ��!       "	�B�iޞk@�B�iޞk@!�B�iޞk@*      ��!       2      ��!       :	��n�@�?��n�@�?!��n�@�?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown� b���?!� b���?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknown�$�=Z�?!�Y�����?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown~����?!��r�e��?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownx��-bۥ?!P��(��?"i
Lgradient_tape/model/block-4_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�)�%�P�?!�������?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown.V�u�?!I��/?��?"?
"model/block-1_conv_0/Conv2D:Conv2DUnknown�������?!<B�a���?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknowns��9~�?!�Z�|��?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknownԔBz�۟?!�C�&:��?"?
"model/block-4_conv_1/Conv2D:Conv2DUnknown%�T�?!G�Ntz��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 