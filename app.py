from datetime import datetime
from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

# Carregar o modelo e o scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter a data selecionada do formulário
    selected_date = request.form['date']
    selected_date = datetime.strptime(selected_date, '%Y-%m-%d')

    # Calcular os dias até a data selecionada
    today = datetime.today()
    dias_ate_a_data = (selected_date - today).days

    # Obter o período do dia (manhã, tarde, noite)
    selected_period = request.form.get('period')

    # Prever a frequência de visitas usando apenas os dias até a data
    input_data = pd.DataFrame([[dias_ate_a_data]], columns=['Dias até a data'])
    scaled_data = scaler.transform(input_data)
    predicted_visits = model.predict(scaled_data)[0]

    # Ajustar a previsão para cada período do dia
    if selected_period == 'manhã':
        period_visits = int(predicted_visits * 0.3)  
    elif selected_period == 'tarde':
        period_visits = int(predicted_visits * 0.5)  
    else:
        period_visits = int(predicted_visits * 0.2)  

    return render_template('index.html',period_visits=period_visits, dias_ate_a_data=dias_ate_a_data,selected_period=selected_period)

if __name__ == '__main__':
    app.run(debug=True) 
