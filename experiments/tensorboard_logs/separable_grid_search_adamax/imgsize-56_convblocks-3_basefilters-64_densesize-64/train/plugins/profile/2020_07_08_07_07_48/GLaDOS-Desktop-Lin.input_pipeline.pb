	�0�{�+7@�0�{�+7@!�0�{�+7@	�F�N;<	@�F�N;<	@!�F�N;<	@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�0�{�+7@� ���?1a��_Y�4@I9��m4 �?Y}�R��c�?*	��n��@2]
&Iterator::Model::FlatMap[0]::Generator�/����?!�[���X@)�/����?1�[���X@:Preprocessing2F
Iterator::Model,E�@
�?!      Y@)�X�+��z?1[\���?:Preprocessing2O
Iterator::Model::FlatMap�d��J��?!J�GAr�X@)�"j��Gi?1S�u(���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?6.2 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� ���?� ���?!� ���?      ��!       "	a��_Y�4@a��_Y�4@!a��_Y�4@*      ��!       2      ��!       :	9��m4 �?9��m4 �?!9��m4 �?B      ��!       J	}�R��c�?}�R��c�?!}�R��c�?R      ��!       Z	}�R��c�?}�R��c�?!}�R��c�?JGPU