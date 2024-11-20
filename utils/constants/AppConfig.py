class AppConfig:
    EPOCHS = 30000
    BATCH_SIZE = 350
    MAX_LEN = 50
    RANDOM_SEED = 42
    PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
    MODEL_FEELINGS_ANALYSIS_PATH='./data/models/feeling_analysis.bin'
    MODEL_DEPRESSION_ANALYSIS_PATH='./data/models/depression.bin'
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
    TEXT_STATISTICS_HISTORY = 'text_statistics_history.csv'
    REFACTOR_FIELDS_NAME = {
                        'Carimbo de data/hora':'data_resposta'
                        ,'Li o Termo acima e concordo':'aceitou'
                        ,'Qual  o seu gênero?':'genero'
                        ,'Qual é a sua faixa etária?':'faixa_etaria'
                        ,'Qual é o seu nível de escolaridade?':'escolaridade'
                        ,'Qual é o seu estado civil?':'estado_civil'
                        ,'Qual é a sua renda familiar mensal?':'renda_familiar_mensal'
                        ,'Qual estado que você reside?':'estado'
                        ,'Você ja foi diagnosticada(o) com depressão?':'possui_depressao'
                        ,'Você acredita que lembranças passadas podem refletir suas decisões atualmente?':'lembrancas_podem_refletir'
                        ,'Você acredita que lembranças da sua infância refletem suas decisões atualmente?':'lembrancas_infancia_podem_refletir'
                        ,'Você consegue identificar suas emoções?':'identifica_emocoes'
                        ,'Você acredita que a prática de atividades físicas podem ajudar no tratamento da saúde mental?':'atividade_fisica_influencia_saude_mental'
                        ,'Você faz terapia com um profissional regularmente?':'faz_terapia_regularmente'
                        ,'Cite até 3 emoções que você consiga lembrar (Separadas por vírgula \',\').':'emocoes_conhecidas'
                        ,'Descreva uma lembrança do seu passado que pode ter refletido em alguma decisão atual.':'descricao_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança na época do ocorrido?':'emocoes_lembranca_passado'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a lembrança atualmente?':'emocoes_lembranca_transformada'
                        ,'Descreva uma situação ocorrida atualmente que pode refletir em suas decisões futuras':'lembranca_atual_futuro'
                        ,'Com base nas emoções listadas abaixo, qual delas se encaixa com a situação no momento em que ocorreu?':'emocoes_lembranca_atual'
                        ,'Com base nas emoções listadas abaixo, qual delas você acredita que pode sentir no futuro quando se lembrar deste ocorrido?':'emocoes_lembranca_atual_transformada_futuro'
                        }