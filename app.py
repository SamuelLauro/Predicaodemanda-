from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado e dados de treinamento
model = joblib.load('modelo_regressao_linear.joblib')
label_encoder = joblib.load('label_encoder.joblib')  # Carregar o LabelEncoder treinado

# Carregar o conjunto de dados de treinamento (se ainda não estiver carregado)
# Certifique-se de que os dados de treinamento foram processados da mesma forma que os dados de entrada
# Aqui você deve ter o dataframe original usado para treinar o modelo
# Por exemplo: df_train = pd.read_csv('dados_treinamento.csv')

# Supondo que 'frequencia' é a coluna que contém as frequências reais de visitas
frequencias_treinamento = np.array([10, 15, 20, 25, 30])  # Exemplo: substitua pelos dados reais
media_frequencia = np.mean(frequencias_treinamento)
percentil_75 = np.percentile(frequencias_treinamento, 75)

# Função para prever o número de clientes com base nos inputs do usuário
def predict_clients(idade, horario_preferido):
    try:
        # Codificar o valor de 'horario_preferido' com o LabelEncoder
        horario_codificado = label_encoder.transform([horario_preferido])[0]
    except ValueError as e:
        print(f"Erro na codificação de '{horario_preferido}': {e}")
        return None
    
    # Preparar os dados de entrada para o modelo
    input_data = pd.DataFrame({
        'ano': [2024],  # Ajustar conforme necessário
        'mês': [10],
        'dia': [17],
        'dia_da_semana': [3],
        'dias_entre_visitas': [30],
        'qual seu horário preferido para agendar um atendimento?': [horario_codificado]
    })

    print(f"Dados de entrada para o modelo: \n{input_data}")  # Linha de depuração
    
    # Fazer a previsão
    previsao = model.predict(input_data)[0]
    print(f"Previsão bruta do modelo: {previsao}")  # Linha de depuração
    
    # Ajustar a previsão com base na média e no percentil
    if previsao < media_frequencia * 0.5:  # Se a previsão for muito baixa
        frequencia_prevista = round(media_frequencia)
    elif previsao > percentil_75:  # Se a previsão for muito alta
        frequencia_prevista = round(percentil_75)
    else:  # Se a previsão estiver dentro do intervalo esperado
        frequencia_prevista = round(previsao)
    
    print(f"Frequência prevista (ajustada): {frequencia_prevista}")  # Linha de depuração
    return frequencia_prevista

@app.route('/', methods=['GET', 'POST'])
def index():
    frequencia_prevista = None
    data_selecionada = None

    if request.method == 'POST':
        data_futura = request.form['data_futura']
        idade = int(request.form['idade'])
        horario_preferido = request.form['horario_preferido']

        frequencia_prevista = predict_clients(idade, horario_preferido)
        data_selecionada = data_futura

    return render_template('index.html', frequencia=frequencia_prevista, data=data_selecionada)

if __name__ == '__main__':
    app.run(debug=True)
