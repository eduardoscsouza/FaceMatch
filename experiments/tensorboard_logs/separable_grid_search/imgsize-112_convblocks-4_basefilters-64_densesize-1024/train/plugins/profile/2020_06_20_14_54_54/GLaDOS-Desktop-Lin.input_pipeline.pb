	닄���[@닄���[@!닄���[@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$닄���[@]�mO�X�?1c�tv24[@Iۣ7�Gn�?*	\����K�@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator4Փ�G�B@!�iX��X@)4Փ�G�B@1�iX��X@:Preprocessing2F
Iterator::Model<�\�g�?!�}�9�?)f��
��?1�d�)�?:Preprocessing2P
Iterator::Model::Prefetch�����j�?!D|�qJ�?)�����j�?1D|�qJ�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapʨ2���B@!�C�1�X@){O崧�l?1dD;!G�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]�mO�X�?]�mO�X�?!]�mO�X�?      ��!       "	c�tv24[@c�tv24[@!c�tv24[@*      ��!       2      ��!       :	ۣ7�Gn�?ۣ7�Gn�?!ۣ7�Gn�?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU