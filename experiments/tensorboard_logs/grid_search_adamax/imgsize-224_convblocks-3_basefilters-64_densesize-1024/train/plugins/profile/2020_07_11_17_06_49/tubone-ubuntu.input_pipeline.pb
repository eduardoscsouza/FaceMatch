	��X�\h@��X�\h@!��X�\h@	�4����?�4����?!�4����?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��X�\h@-�1 �?1�gx��g@I����X��?Y5E�ӻ@*	��"��]�@2]
&Iterator::Model::FlatMap[0]::GeneratorH�9���@!quU��X@)H�9���@1quU��X@:Preprocessing2F
Iterator::Model=��@f�@!      Y@)�?�@�w�?1X�&(�?:Preprocessing2O
Iterator::Model::FlatMap��e���@!�� ���X@)�i�WV�t?1�I�V}�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	-�1 �?-�1 �?!-�1 �?      ��!       "	�gx��g@�gx��g@!�gx��g@*      ��!       2      ��!       :	����X��?����X��?!����X��?B      ��!       J	5E�ӻ@5E�ӻ@!5E�ӻ@R      ��!       Z	5E�ӻ@5E�ӻ@!5E�ӻ@JGPU