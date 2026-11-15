import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

EXPERIMENT_ID = "exp_nslkdd_mlp_fgsm_v1"
EXPERIMENT_DESCRIPTION = (
    "Эксперимент по атаке FGSM (ART) на MLP-классификатор бинарной "
    "детекции сетевых вторжений на датасете NSL-KDD."
)

AI_SYSTEM_ID = "model_nslkdd_mlp_v1"
EVALUATION_METHOD_ID = "art_fast_gradient_method"
APPLICATION_ENVIRONMENT_ID = "env_py310_tf213_art117"
LIFE_CYCLE_STAGE_ID = "testing"
OBJECT_DOMAIN_ID = "network_intrusion_detection"

FGM_EPS = 0.1
EPOCHS = 15
BATCH_SIZE = 256
RANDOM_SEEDS = [42, 43, 44]

MODEL_PATH = os.path.join(MODELS_DIR, "nslkdd_mlp.keras")
RESULTS_PATH = os.path.join(RESULTS_DIR, f"results_{EXPERIMENT_ID}.json")

DATASET_INFO = {
    "name": "NSL-KDD",
    "version": "KDDTrain+ / KDDTest+ (original)",
    "source": "https://www.unb.ca/cic/datasets/nsl.html",
    "train_file": "KDDTrain+.txt",
    "test_file": "KDDTest+.txt",
}
MODEL_INFO = {
    "name": "MLP binary classifier",
    "version": "1.0",
    "framework": "TensorFlow/Keras",
    "source": "trained in this project",
    "architecture": "Dense(64,relu) -> Dense(32,relu) -> Dense(2,softmax)",
}
ATTACK_INFO = {
    "library": "adversarial-robustness-toolbox",
    "method": "FastGradientMethod",
    "params": {"eps": FGM_EPS, "norm": "inf"},
}
LIFE_CYCLE_STAGE_DESCRIPTION = (
    "Стадия testing: оценка устойчивости обученной модели "
    "к состязательным возмущениям на тестовой выборке."
)

RETRAIN_PER_RUN = True
PROJECT_ONEHOT = True
ROUND_BINARY = True
SAVE_PREDICTIONS = True

SAVE_TRAINING_CSV = True
SAVE_TRAINING_TXT = True
SAVE_RUN_LOG = True
