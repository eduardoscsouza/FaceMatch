	�;2V�`@�;2V�`@!�;2V�`@	��a@��a@!��a@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�;2V�`@,�-Xj�?1��;���_@I��N����?Yګ����	@*	�I+s�@2]
&Iterator::Model::FlatMap[0]::Generator�`7l[@!�Q9�<�X@)�`7l[@1�Q9�<�X@:Preprocessing2F
Iterator::Model�M+�@.@!      Y@)�,��o�?1�>LW�b�?:Preprocessing2O
Iterator::Model::FlatMap� v��@!���2��X@)�� ���e?1pȈ���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,�-Xj�?,�-Xj�?!,�-Xj�?      ��!       "	��;���_@��;���_@!��;���_@*      ��!       2      ��!       :	��N����?��N����?!��N����?B      ��!       J	ګ����	@ګ����	@!ګ����	@R      ��!       Z	ګ����	@ګ����	@!ګ����	@JGPU