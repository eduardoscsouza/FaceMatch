	Z�H��(w@Z�H��(w@!Z�H��(w@	ȿ�X�@ȿ�X�@!ȿ�X�@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Z�H��(w@��H���?1K �)�tv@I��b��?Y�k	��?#@*	�V>�@2]
&Iterator::Model::FlatMap[0]::Generator�	�_�v#@!B1���X@)�	�_�v#@1B1���X@:Preprocessing2F
Iterator::Model$�jf-}#@!      Y@)��y���?1J�ИѶ�?:Preprocessing2O
Iterator::Model::FlatMap��&�x#@!�˙KR�X@)�j,am�m?1��4��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��H���?��H���?!��H���?      ��!       "	K �)�tv@K �)�tv@!K �)�tv@*      ��!       2      ��!       :	��b��?��b��?!��b��?B      ��!       J	�k	��?#@�k	��?#@!�k	��?#@R      ��!       Z	�k	��?#@�k	��?#@!�k	��?#@JGPU