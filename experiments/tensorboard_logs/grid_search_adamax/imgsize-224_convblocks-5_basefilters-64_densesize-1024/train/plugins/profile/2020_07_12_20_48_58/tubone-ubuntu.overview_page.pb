�	s��/ik@s��/ik@!s��/ik@	:�Y	r@:�Y	r@!:�Y	r@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-s��/ik@	�^)K�?1ퟧzj@I��ʡE�?Y�V`�j@*	�G�z�;�@2]
&Iterator::Model::FlatMap[0]::Generator@�:s�@!�����X@)@�:s�@1�����X@:Preprocessing2F
Iterator::Model1��PN�@!      Y@)��R�?1�j�M��?:Preprocessing2O
Iterator::Model::FlatMap.��'H�@!�J=Y0�X@))w���i?1���4�L�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
		�^)K�?	�^)K�?!	�^)K�?      ��!       "	ퟧzj@ퟧzj@!ퟧzj@*      ��!       2      ��!       :	��ʡE�?��ʡE�?!��ʡE�?B      ��!       J	�V`�j@�V`�j@!�V`�j@R      ��!       Z	�V`�j@�V`�j@!�V`�j@JGPU�"i
Lgradient_tape/model/block-0_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�]q��?!�]q��?"i
Lgradient_tape/model/block-4_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknownJgKn|��?!Ј˜"��?"?
"model/block-0_conv_1/Conv2D:Conv2DUnknown� �uf��?!�8:�$�?"g
Jgradient_tape/model/block-0_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknownۿi�R�?!���Ь�?"i
Lgradient_tape/model/block-1_conv_1/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�S��=a�?!��F2��?"?
"model/block-4_conv_1/Conv2D:Conv2DUnknown�"��?!�b����?"T
+Adamax/Adamax/update_20/ResourceApplyAdaMaxResourceApplyAdaMax1�9,o��?!ԗ��/�?"?
"model/block-1_conv_1/Conv2D:Conv2DUnknown�D�`�?!n`�8�;�?"g
Jgradient_tape/model/block-4_conv_1/Conv2DBackpropInput:Conv2DBackpropInputUnknown�;�a�Y�?!�g%�F�?"i
Lgradient_tape/model/block-1_conv_0/Conv2DBackpropFilter:Conv2DBackpropFilterUnknown�!�n��?!���(�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 