class AppConfig:
    EPOCHS = 100
    BATCH_SIZE = 100
    MAX_LEN = 160
    RANDOM_SEED = 666
    PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
    MODEL_FEELINGS_ANALYSIS_PATH='./data/models/feeling_analysis.bin'
    FEELINGS_ANALYSIS_CLASSES = ['negativo', 'neutro', 'positivo']
    FILE_DATA = 'dados_iniciais.csv'
    DIRECTORY_DATA = './dados/'
    EMOTIONS_FILE = 'emocoes.csv'
    DATA_FINAL='dados.csv'
    WRANGLED_DATA_FINAL = 'out_dados.csv' 
    TRAIN_PERCENTAGE = 0.7
    TEST_PERCENTAGE = 0.2
    LEARNING_RATE=3e-5