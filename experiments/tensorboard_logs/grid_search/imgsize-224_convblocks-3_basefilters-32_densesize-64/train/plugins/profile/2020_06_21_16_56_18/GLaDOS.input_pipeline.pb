	�!�̠:�@�!�̠:�@!�!�̠:�@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�!�̠:�@u�׃Iq�?1
��.�@I��r�S��?*	Q��n��@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�J�({yN@!QH��X@)�J�({yN@1QH��X@:Preprocessing2F
Iterator::Model��ao�?!P^�&�M�?)΋_�(�?1���B)�?:Preprocessing2P
Iterator::Model::Prefetch#�	��?!��!FP��?)#�	��?1��!FP��?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap׽�	zN@!)_v�,�X@)��j�q?1:ε�+}?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	u�׃Iq�?u�׃Iq�?!u�׃Iq�?      ��!       "	
��.�@
��.�@!
��.�@*      ��!       2      ��!       :	��r�S��?��r�S��?!��r�S��?B      ��!       J      ��!       R      ��!       Z      ��!       JGPU