	��p>mF@��p>mF@!��p>mF@	={�	�5P@={�	�5P@!={�	�5P@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��p>mF@���R���?1jO�9��,@IA.q���?Yzލ�=@*	�ʡEƌ�@2]
&Iterator::Model::FlatMap[0]::Generator�"h�$:=@!K;�A�X@)�"h�$:=@1K;�A�X@:Preprocessing2F
Iterator::Model�Z�a/<=@!      Y@)�5��x?1dwrY��?:Preprocessing2O
Iterator::Model::FlatMap���:=@!�hj��X@)���V%a?1�Q���R}?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 64.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���R���?���R���?!���R���?      ��!       "	jO�9��,@jO�9��,@!jO�9��,@*      ��!       2      ��!       :	A.q���?A.q���?!A.q���?B      ��!       J	zލ�=@zލ�=@!zލ�=@R      ��!       Z	zލ�=@zލ�=@!zލ�=@JGPU