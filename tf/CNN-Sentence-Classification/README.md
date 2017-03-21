# Convolutional Neural Networks for Sentence Classification(Sentiment Analysis)

The project is a simplified implemention of [ Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) in Tensorflow with reference to [dennybritz's blog](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/),which is a really nice tutorial for CNN applied to NLP.

Author: Lebron Ran
Language: Python

# Training
> python train_CNN4Text.py  --help

output:

	usage: train_CNN4Text.py [-h]
                         [--validation_set_percentage VALIDATION_SET_PERCENTAGE]
                         [--data_postive_path DATA_POSTIVE_PATH]
                         [--data_negative_path DATA_NEGATIVE_PATH]
                         [--learning_rate LEARNING_RATE]
                         [--embedding_size EMBEDDING_SIZE]
                         [--num_filters NUM_FILTERS]
                         [--filter_sizes FILTER_SIZES] [--keep_prob KEEP_PROB]
                         [--l2_reg_lambda L2_REG_LAMBDA]
                         [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                         [--evaluate_interval EVALUATE_INTERVAL]
                         [--checkpoint_interval CHECKPOINT_INTERVAL]
                         [--num_checkpoints NUM_CHECKPOINTS]
                         [--allow_soft_parameters [ALLOW_SOFT_PARAMETERS]]
                         [--noallow_soft_parameters]
                         [--log_device_placement [LOG_DEVICE_PLACEMENT]]
                         [--nolog_device_placement]


# Evaluating

> python evaluate.py --checkpoint_path=/path/to/checkpoints
