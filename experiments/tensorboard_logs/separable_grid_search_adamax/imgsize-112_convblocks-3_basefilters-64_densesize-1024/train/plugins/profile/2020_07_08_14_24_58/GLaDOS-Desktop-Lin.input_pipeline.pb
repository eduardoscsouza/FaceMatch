	i�[@i�[@!i�[@	Q�����?Q�����?!Q�����?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-i�[@5S��?1=}��KZ@Iam����?Y2���j�?*	L7�A`��@2]
&Iterator::Model::FlatMap[0]::Generator�F��?�?!�Gl��X@)�F��?�?1�Gl��X@:Preprocessing2F
Iterator::Model��C p�?!      Y@)�ю~7}?1�)(���?:Preprocessing2O
Iterator::Model::FlatMap�O ��R�?!��z�/�X@)�Ϲ��r?1��,�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	5S��?5S��?!5S��?      ��!       "	=}��KZ@=}��KZ@!=}��KZ@*      ��!       2      ��!       :	am����?am����?!am����?B      ��!       J	2���j�?2���j�?!2���j�?R      ��!       Z	2���j�?2���j�?!2���j�?JGPU