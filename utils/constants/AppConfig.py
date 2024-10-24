class AppConfig:
    EPOCHS = 300
    BATCH_SIZE = 188
    MAX_LEN = 100
    RANDOM_SEED = 42
    PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
    MODEL_FEELINGS_ANALYSIS_PATH='./data/models/feeling_analysis.bin'
    MODEL_DEPRESSION_PATH='./data/models/depression.bin'
    FEELINGS_ANALYSIS_CLASSES = ['negativo', 'neutro', 'positivo']
    PREDICTED_DEPRESSION_FILE = 'depression_test_predicted.csv'
    FILE_DATA = 'dados_iniciais.csv'
    DIRECTORY_DATA = './data/files/'
    EMOTIONS_FILE = 'emocoes.csv'
    DATA_FINAL='dados.csv'
    WRANGLED_DATA_FINAL = 'out_dados.csv' 
    TRAIN_PERCENTAGE = 0.7
    TEST_PERCENTAGE = 0.2
    LEARNING_RATE=3e-5