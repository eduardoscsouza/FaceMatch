�	��9]+)@��9]+)@!��9]+)@	�%s�.-@�%s�.-@!�%s�.-@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��9]+)@IG9�M��?1Eh�?"@I#�W<�H�?Y��J"�`�?*	�ZdZ�@2]
&Iterator::Model::FlatMap[0]::Generatorx{���?!/�4\�X@)x{���?1/�4\�X@:Preprocessing2F
Iterator::Model�K�Ƽ�?!      Y@)T㥛� �?1��$���?:Preprocessing2O
Iterator::Model::FlatMap �={��?!+�@�+�X@): 	�va?1�'1(f�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 14.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"@11.1 % of the total step time sampled is spent on Kernel Launch.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	IG9�M��?IG9�M��?!IG9�M��?      ��!       "	Eh�?"@Eh�?"@!Eh�?"@*      ��!       2      ��!       :	#�W<�H�?#�W<�H�?!#�W<�H�?B      ��!       J	��J"�`�?��J"�`�?!��J"�`�?R      ��!       Z	��J"�`�?��J"�`�?!��J"�`�?JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown3�5܁B�?!3�5܁B�?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�-�u��?!|"�N8�?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown��نs�?!>��&�J�?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknown;d� n9�?!ş��ő�?"i
Lgradient_tape/model/block-0_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown��n��-�?!s}�{w�?"-
IteratorGetNext/_1_Send�U�����?!ɝ�
R�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�9��?!��s�?"i
Lgradient_tape/model/block-2_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownyG�q�?!$�u����?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknownVJ��E�?!9��v��?"g
Jgradient_tape/model/block-1_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown��Љ�c�?!�г�'��?2blackQ      Y@"�
both�Your program is MODERATELY input-bound because 14.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate@11.1 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 