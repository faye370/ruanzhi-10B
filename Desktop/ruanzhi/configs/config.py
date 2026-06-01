MODEL_NAME = "bert-base-uncased"
DATASET = "imdb"
NUM_LABELS = 2
MAX_LENGTH = 256

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
CHECKPOINT_DIR = "./checkpoints/bert-imdb"

# Pre-trained BERT fine-tuned on IMDB (TextAttack official, acc=93.7%).
# Use this to skip local training when no GPU is available.
PRETRAINED_MODEL_DIR = "textattack/bert-base-uncased-imdb"

ATTACK_NUM_EXAMPLES = 1000
RESULTS_DIR = "./results"
