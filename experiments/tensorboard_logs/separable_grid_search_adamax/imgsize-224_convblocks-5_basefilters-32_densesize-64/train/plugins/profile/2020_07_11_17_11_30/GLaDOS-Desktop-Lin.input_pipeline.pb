	�x]��Gt@�x]��Gt@!�x]��Gt@	>%�(@q@>%�(@q@!>%�(@q@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�x]��Gt@EKOˏ�?1��!��s@I�5"�?Y
�B���#@*	�~j�7�@2]
&Iterator::Model::FlatMap[0]::Generatori��U�$@!�u�Zz�X@)i��U�$@1�u�Zz�X@:Preprocessing2F
Iterator::Model�Ƕ8�$@!      Y@)'�����?1��G�1w�?:Preprocessing2O
Iterator::Model::FlatMap@�J���$@!n�3b�X@)��.5B?s?1�z�/�>�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	EKOˏ�?EKOˏ�?!EKOˏ�?      ��!       "	��!��s@��!��s@!��!��s@*      ��!       2      ��!       :	�5"�?�5"�?!�5"�?B      ��!       J	
�B���#@
�B���#@!
�B���#@R      ��!       Z	
�B���#@
�B���#@!
�B���#@JGPU