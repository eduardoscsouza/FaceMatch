	B��^~@B��^~@!B��^~@	�0�U�N.@�0�U�N.@!�0�U�N.@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-B��^~@v�U�0��?1�S���~@Iw0b� � @Y<�(A!�?*	V-�a�@2]
&Iterator::Model::FlatMap[0]::Generator��Z�[��?!Q�X5k�X@)��Z�[��?1Q�X5k�X@:Preprocessing2F
Iterator::Modelm��}��?!      Y@)��$��|?1���Ѷ�?:Preprocessing2O
Iterator::Model::FlatMap���jد�?!��\���X@)�2nj��l?1�G�]Y�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 15.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"@34.7 % of the total step time sampled is spent on Kernel Launch.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	v�U�0��?v�U�0��?!v�U�0��?      ��!       "	�S���~@�S���~@!�S���~@*      ��!       2      ��!       :	w0b� � @w0b� � @!w0b� � @B      ��!       J	<�(A!�?<�(A!�?!<�(A!�?R      ��!       Z	<�(A!�?<�(A!�?!<�(A!�?JGPU