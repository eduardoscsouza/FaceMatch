	8� "5�[@8� "5�[@!8� "5�[@	�Ώ���@�Ώ���@!�Ώ���@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-8� "5�[@�W�ۼ�?1�({K9�Z@I)�7�E�?Y�(�[Z�@*	�(\�B��@2]
&Iterator::Model::FlatMap[0]::Generator�X�@!��'��X@)�X�@1��'��X@:Preprocessing2F
Iterator::Model��"�-'@!      Y@)�uoEb�z?1]>k����?:Preprocessing2O
Iterator::Model::FlatMapC p�@!aJ� ��X@)��N�jp?1wt	���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�W�ۼ�?�W�ۼ�?!�W�ۼ�?      ��!       "	�({K9�Z@�({K9�Z@!�({K9�Z@*      ��!       2      ��!       :	)�7�E�?)�7�E�?!)�7�E�?B      ��!       J	�(�[Z�@�(�[Z�@!�(�[Z�@R      ��!       Z	�(�[Z�@�(�[Z�@!�(�[Z�@JGPU