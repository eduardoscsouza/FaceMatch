	��֦1�u@��֦1�u@!��֦1�u@	��z�?�@��z�?�@!��z�?�@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��֦1�u@m�����?1FaEOu@I�
G�J��?Y�>��Ȫ$@*	�I{��@2]
&Iterator::Model::FlatMap[0]::Generator�ެ�C%@!�]��h�X@)�ެ�C%@1�]��h�X@:Preprocessing2F
Iterator::ModeldT8J%@!      Y@)���G�Ȁ?1 �-PF��?:Preprocessing2O
Iterator::Model::FlatMap�hr1F%@!��k��X@)��v�$$r?1����sM�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	m�����?m�����?!m�����?      ��!       "	FaEOu@FaEOu@!FaEOu@*      ��!       2      ��!       :	�
G�J��?�
G�J��?!�
G�J��?B      ��!       J	�>��Ȫ$@�>��Ȫ$@!�>��Ȫ$@R      ��!       Z	�>��Ȫ$@�>��Ȫ$@!�>��Ȫ$@JGPU