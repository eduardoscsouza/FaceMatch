	y[�يI@y[�يI@!y[�يI@	����$@����$@!����$@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-y[�يI@��֥�?1���w��E@IV�f���?Y�C�Hp@*	ffffF9�@2]
&Iterator::Model::FlatMap[0]::Generator�}��Ź@!�w���X@)�}��Ź@1�w���X@:Preprocessing2F
Iterator::Model��j+��@!      Y@)p}Xo�
�?19[cm��?:Preprocessing2O
Iterator::Model::FlatMap�;3�p�@!RNɧ��X@)�r��h�r?1{Z�`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 10.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��֥�?��֥�?!��֥�?      ��!       "	���w��E@���w��E@!���w��E@*      ��!       2      ��!       :	V�f���?V�f���?!V�f���?B      ��!       J	�C�Hp@�C�Hp@!�C�Hp@R      ��!       Z	�C�Hp@�C�Hp@!�C�Hp@JGPU