import argparse
# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = './vqa_api/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = './vqa_api/PythonEvaluationTools'

# location of the data
VQA_PREFIX = './vqa_api/'

feat = 'faster_rcnn_resnet_pool5'
DATA_PATHS = {
    'train': {
        'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_train2014_questions.json',
        'ans_file': VQA_PREFIX + '/Annotations/v2_mscoco_train2014_annotations.json',
        'features_prefix': VQA_PREFIX + '/Features/%s/trainval_36/COCO_trainval_'%feat
        # 'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/train2014/COCO_train2014_'%feat
    },
    'val': {
        'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_val2014_questions.json',
        'ans_file': VQA_PREFIX + '/Annotations/v2_mscoco_val2014_annotations.json',
        'features_prefix': VQA_PREFIX + '/Features/%s/trainval_36/COCO_trainval_'%feat
        # 'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/val2014/COCO_val2014_'%feat
    },
    'test-dev': {
        'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json',
        'features_prefix': VQA_PREFIX + '/Features/%s/test2015_36/COCO_test_'%feat
        # 'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
    },
    'test': {
        'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_test2015_questions.json',
        'features_prefix': VQA_PREFIX + '/Features/%s/test2015_36/COCO_test_'%feat
        # 'features_prefix': VQA_PREFIX + '/Features/ms_coco/resnet_%s_bgrms_large/test2015/COCO_test2015_'%feat
    },
    'genome': {
        'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
        # 'features_prefix': VQA_PREFIX + '/Features/genome/feat_resnet-152/resnet_%s_bgrms_large/'%feat
    }
}
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--TRAIN_GPU_ID', type=int, default=0)
    parser.add_argument('--TEST_GPU_ID', type=int, default=0)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=64)
    parser.add_argument('--NUM_OUTPUT_UNITS', type=int, default=3000)
    parser.add_argument('--MAX_WORDS_IN_QUESTION', type=int, default=15)
    parser.add_argument('--MAX_ITERATIONS', type=int, default=200000)
    parser.add_argument('--PRINT_INTERVAL', type=int, default=100)
    parser.add_argument('--TESTDEV_INTERVAL', type=int, default=100000)
    parser.add_argument('--CHECKPOINT_INTERVAL', type=int, default=5000)
    parser.add_argument('--RESUME', type=bool, default=False)
    parser.add_argument('--RESUME_PATH', type=str, default='./data/drau_glove_iter_100000.pth')
    parser.add_argument('--VAL_INTERVAL', type=int, default=100000)
    parser.add_argument('--IMAGE_CHANNEL', type=int, default=2048)
    parser.add_argument('--INIT_LERARNING_RATE', type=float, default=0.001)
    parser.add_argument('--DECAY_STEPS', type=int, default=50000)
    parser.add_argument('--DECAY_RATE', type=float, default=0.5)    # TODO: fix naming. this is LR decay not weight
    parser.add_argument('--TRAIN_DATA_SPLITS', type=str, default='train')
    parser.add_argument('--QUESTION_VOCAB_SPACE', type=str, default='train')
    parser.add_argument('--ANSWER_VOCAB_SPACE', type=str, default='train')

    parser.add_argument('--NUM_IMG_GLIMPSE', type=int, default=2)
    parser.add_argument('--JOINT_EMBEDDING_SIZE', type=int, default=1024)
    parser.add_argument('--NUM_QUESTION_GLIMPSE', type=int, default=2)
    parser.add_argument('--IMG_FEAT_SIZE', type=int, default=36)
    parser.add_argument('--FUSION_OUT_DIM', type=int, default=16000)
    parser.add_argument('--LSTM_UNIT_NUM', type=int, default=1024)
    parser.add_argument('--LSTM_DROPOUT_RATIO', type=float, default=0.3)
    parser.add_argument('--FUSION_DROPOUT_RATIO', type=float, default=0.1)
     
    parser.add_argument('--IMG_FEAT_TYPE', type=str, default=feat)
   
    args = parser.parse_args([])
    return args
