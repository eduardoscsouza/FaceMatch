	|a2UjM@|a2UjM@!|a2UjM@	U�l��;�?U�l��;�?!U�l��;�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-|a2UjM@PP�V��?1{���L@I+ٱ���?Y/m8,��?*	�l��W�@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�R\U�@@!��v��X@)�R\U�@@1��v��X@:Preprocessing2F
Iterator::Model�>���?!��1��?)+�� �?1�a)܆��?:Preprocessing2P
Iterator::Model::Prefetch��5�e�?!_t )�?)��5�e�?1_t )�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapSAEկ@@!�(y��X@)ap��/w?1���\#�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	PP�V��?PP�V��?!PP�V��?      ��!       "	{���L@{���L@!{���L@*      ��!       2      ��!       :	+ٱ���?+ٱ���?!+ٱ���?B      ��!       J	/m8,��?/m8,��?!/m8,��?R      ��!       Z	/m8,��?/m8,��?!/m8,��?JGPU