	^f���t@^f���t@!^f���t@	��01@��01@!��01@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-^f���t@����?1\�d8�$t@IeM.�@�?Y���4$@*	�t����@2]
&Iterator::Model::FlatMap[0]::Generator����K$@!y�8���X@)����K$@1y�8���X@:Preprocessing2F
Iterator::Model��@��Q$@!      Y@)� !���?1!����?:Preprocessing2O
Iterator::Model::FlatMapTrN�M$@!Yy����X@)�]K�=k?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����?����?!����?      ��!       "	\�d8�$t@\�d8�$t@!\�d8�$t@*      ��!       2      ��!       :	eM.�@�?eM.�@�?!eM.�@�?B      ��!       J	���4$@���4$@!���4$@R      ��!       Z	���4$@���4$@!���4$@JGPU