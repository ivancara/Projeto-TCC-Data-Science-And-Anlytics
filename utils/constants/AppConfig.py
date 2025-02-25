class AppConfig:
    EPOCHS = 30000
    BATCH_SIZE = 350
    MAX_LEN = 50
    RANDOM_SEED = 42
    MODEL_FEELINGS_ANALYSIS_PATH='./data/models/feeling_analysis.bin'
    MODEL_DEPRESSION_ANALYSIS_PATH='./data/models/depression.bin'
    PREDICTED_DEPRESSION_FILE = 'depression_test_predicted.csv'
    FILE_DATA = 'dados_iniciais.csv'
    DIRECTORY_DATA = './data/files/'
    EMOTIONS_FILE = 'emocoes.csv'
    DATA_FINAL='dados.csv'
    WRANGLED_DATA_FINAL = 'out_dados.csv' 
    TRAIN_PERCENTAGE = 0.7
    TEST_PERCENTAGE = 0.2
    LEARNING_RATE=3e-5
    CORRELATION='pearson'
    REFACTOR_FIELDS_NAME = {
                        'Carimbo de data/hora':'data_resposta'
                        ,'Qual  o seu gênero?':'genero'
                        ,'Qual é a sua faixa etária?':'faixa_etaria'
                        ,'Qual é o seu nível de escolaridade?':'escolaridade'
                        ,'Qual é o seu estado civil?':'estado_civil'
                        ,'Qual é a sua renda familiar mensal?':'renda_familiar_mensal'
                        ,'Você ja foi diagnosticada(o) com depressão?':'possui_depressao'
                        ,'Você acredita que lembranças passadas podem refletir suas decisões atualmente?':'lembrancas_podem_refletir'
                        ,'Você acredita que lembranças da sua infância refletem suas decisões atualmente?':'lembrancas_infancia_podem_refletir'
                        ,'Você consegue identificar suas emoções?':'identifica_emocoes'
                        ,'Você acredita que a prática de atividades físicas podem ajudar no tratamento da saúde mental?':'atividade_fisica_influencia_saude_mental'
                        ,'Você faz terapia com um profissional regularmente?':'faz_terapia_regularmente'
                        ,'Cite até 3 emoções que você consiga lembrar (Separadas por vírgula \',\').':'emocoes_conhecidas'
                        }