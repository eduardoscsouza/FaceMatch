	>���nx@>���nx@!>���nx@	�iJ"��@�iJ"��@!�iJ"��@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails->���nx@ޑ���?�?1�)"aw@IQ���x�?YC p�1#@*	�x�&qS�@2]
&Iterator::Model::FlatMap[0]::Generator��-@�#@!�~"�>�X@)��-@�#@1�~"�>�X@:Preprocessing2F
Iterator::ModelemS<.�#@!      Y@)JB"m�O�?1�+H騹?:Preprocessing2O
Iterator::Model::FlatMap�$xC�#@!���ŕ�X@)+N�f�m?1ڭx�<��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ޑ���?�?ޑ���?�?!ޑ���?�?      ��!       "	�)"aw@�)"aw@!�)"aw@*      ��!       2      ��!       :	Q���x�?Q���x�?!Q���x�?B      ��!       J	C p�1#@C p�1#@!C p�1#@R      ��!       Z	C p�1#@C p�1#@!C p�1#@JGPU