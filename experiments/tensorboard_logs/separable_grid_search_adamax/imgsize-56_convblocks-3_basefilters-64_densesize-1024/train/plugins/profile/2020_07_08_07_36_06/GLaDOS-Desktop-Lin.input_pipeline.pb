	_\��+9@_\��+9@!_\��+9@	�q���@�q���@!�q���@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-_\��+9@���7�{�?1�:�W6@I%[]N	��?Y�I��	�?*	���S�z�@2]
&Iterator::Model::FlatMap[0]::Generator�۽�'��?!�&i֕�X@)�۽�'��?1�&i֕�X@:Preprocessing2F
Iterator::ModelhB�Ē��?!      Y@)������}?1����1��?:Preprocessing2O
Iterator::Model::FlatMapᶶ���?!G~望�X@)!���'*k?1'�W}��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?5.4 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���7�{�?���7�{�?!���7�{�?      ��!       "	�:�W6@�:�W6@!�:�W6@*      ��!       2      ��!       :	%[]N	��?%[]N	��?!%[]N	��?B      ��!       J	�I��	�?�I��	�?!�I��	�?R      ��!       Z	�I��	�?�I��	�?!�I��	�?JGPU