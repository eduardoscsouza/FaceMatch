	�ʉv�X@�ʉv�X@!�ʉv�X@	Ei�X�@Ei�X�@!Ei�X�@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ʉv�X@w+Kt�Y�?1��p�Q�V@I�hq�0��?YNA~6R@*	ˡE����@2]
&Iterator::Model::FlatMap[0]::Generator�8EGr)@!P�֎�X@)�8EGr)@1P�֎�X@:Preprocessing2F
Iterator::Modeli�8@!      Y@)t|�8c��?1����l�?:Preprocessing2O
Iterator::Model::FlatMap+�w�7.@!y�I�X@)$��Ps?1.����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w+Kt�Y�?w+Kt�Y�?!w+Kt�Y�?      ��!       "	��p�Q�V@��p�Q�V@!��p�Q�V@*      ��!       2      ��!       :	�hq�0��?�hq�0��?!�hq�0��?B      ��!       J	NA~6R@NA~6R@!NA~6R@R      ��!       Z	NA~6R@NA~6R@!NA~6R@JGPU