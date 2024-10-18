class AppConfig:
    EPOCHS = 10
    BATCH_SIZE = 16
    MAX_LEN = 254
    RANDOM_SEED = 42
    PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
    WRANGLED_DATA_FINAL = 'wrangled_data_final.csv' 
    MODEL_FEELINGS_ANALYSIS_PATH='best_model_state.bin'