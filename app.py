from datetime import datetime, timedelta
from flask import Flask, render_template, request
import joblib
import pandas as pd
import random

app = Flask(__name__)

# Carregar o modelo, o scaler e os nomes das features
model = joblib.load('xgboost_regressor_model.joblib')
scaler = joblib.load('scaler.joblib')
with open('feature_names.joblib', 'rb') as f:
    feature_names = joblib.load(f)

@app.route('/')
def index():
    # Define a data mínima (hoje) e a data máxima (10 dias no futuro)
    min_date = datetime.today().strftime('%Y-%m-%d')
    max_date = (datetime.today() + timedelta(days=10)).strftime('%Y-%m-%d')
    return render_template('index.html', min_date=min_date, max_date=max_date)

@app.route('/predict', methods=['POST'])
def predict():
    # Obter a data selecionada do formulário
    selected_date = request.form['date']
    selected_date = datetime.strptime(selected_date, '%Y-%m-%d')

    # Calcular os dias até a data selecionada
    today = datetime.today()
    dias_ate_a_data = (selected_date - today).days

    # Verificar se a data é no passado ou maior que 10 dias no futuro
    if dias_ate_a_data < 0 or dias_ate_a_data > 10:
        min_date = today.strftime('%Y-%m-%d')
        max_date = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        return render_template('index.html', error="A data selecionada deve estar entre hoje e 10 dias no futuro.", min_date=min_date, max_date=max_date)

    # Obter o período do dia (manhã, tarde, noite)
    selected_period = request.form.get('period')

    # Codificar o período como numérico
    if selected_period == 'manhã':
        horario_preferido = 0
    elif selected_period == 'tarde':
        horario_preferido = 1
    else:  # 'noite'
        horario_preferido = 2

    # Criação do DataFrame de entrada para previsão
    input_data = pd.DataFrame([[dias_ate_a_data, horario_preferido, 10]],  # Exemplo de 10 dias desde a última visita
                              columns=['Dia_da_Semana', 'Turno', 'Dias_Desde_Ultima_Visita'])

    # Adicionar colunas faltantes com valor zero
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reordenar as colunas para garantir a compatibilidade
    input_data = input_data[feature_names]

    # Normalizar os dados de entrada
    scaled_data = scaler.transform(input_data)

    # Prever a frequência de visitas
    predicted_visits = model.predict(scaled_data)[0]

    # Adicionar variabilidade artificial para teste
    predicted_visits += random.uniform(-1, 1)  # Adiciona um valor aleatório entre -1 e 1 para teste

    # Adicionar uma impressão para depuração
    print("Dados de entrada (não escalados):", input_data)
    print("Dados de entrada (escalados):", scaled_data)
    print("Previsão original (sem aleatoriedade):", model.predict(scaled_data)[0])
    print("Previsão com variabilidade (para teste):", predicted_visits)

    return render_template(
        'index.html', 
        predicted_visits=int(predicted_visits), 
        dias_ate_a_data=dias_ate_a_data, 
        selected_period=selected_period,
        error=None,
        min_date=today.strftime('%Y-%m-%d'),
        max_date=(today + timedelta(days=10)).strftime('%Y-%m-%d')
    )

if __name__ == '__main__':
    app.run(debug=True)
