	GUD��T@GUD��T@!GUD��T@	��
%@��
%@!��
%@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-GUD��T@M��΢w�?1�Qd��QR@I
�Rς��?Y֌r�!@*	���k��@2]
&Iterator::Model::FlatMap[0]::GeneratorΊ��>'"@!�lj�X@)Ί��>'"@1�lj�X@:Preprocessing2F
Iterator::Model�ٲ|-"@!      Y@)�/�^|�~?18�"NA1�?:Preprocessing2O
Iterator::Model::FlatMap�&M��)"@!Rw����X@)�`�$�s?1���BM�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 10.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M��΢w�?M��΢w�?!M��΢w�?      ��!       "	�Qd��QR@�Qd��QR@!�Qd��QR@*      ��!       2      ��!       :	
�Rς��?
�Rς��?!
�Rς��?B      ��!       J	֌r�!@֌r�!@!֌r�!@R      ��!       Z	֌r�!@֌r�!@!֌r�!@JGPU