	��Z�{gG@��Z�{gG@!��Z�{gG@	���X@���X@!���X@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��Z�{gG@���*���?1��<,�E@IMHk:!�?Y����ə�?*	�G�za�@2]
&Iterator::Model::FlatMap[0]::Generator���h ��?!K�'!��X@)���h ��?1K�'!��X@:Preprocessing2F
Iterator::Model#-��#�?!      Y@)/�o��e�?1E�X�5��?:Preprocessing2O
Iterator::Model::FlatMap�MbX��?!�W�	�X@)M�O�d?1��_R��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���*���?���*���?!���*���?      ��!       "	��<,�E@��<,�E@!��<,�E@*      ��!       2      ��!       :	MHk:!�?MHk:!�?!MHk:!�?B      ��!       J	����ə�?����ə�?!����ə�?R      ��!       Z	����ə�?����ə�?!����ə�?JGPU