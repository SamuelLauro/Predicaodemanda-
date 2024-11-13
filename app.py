from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from prophet import Prophet


# Inicializar o app Flask
app = Flask(__name__)

# Carregar e preparar os dados
df = pd.read_csv("frequency_data.csv", parse_dates=['date'])
df = df.set_index('date').resample('D').size().rename('clientes').to_frame()
df = df[df['clientes'] > 0].reset_index().rename(columns={'date': 'ds', 'clientes': 'y'})

# Configurar e treinar o modelo Prophet
modelo = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
modelo.fit(df)

# Página inicial com o formulário
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint para fazer a previsão
@app.route('/prever', methods=['POST'])
def prever():
    try:
        # Receber a data escolhida pelo usuário
        data_escolhida = request.form['data']
        data_escolhida = pd.to_datetime(data_escolhida)

        # Validar se a data é no futuro
        ultima_data = df['ds'].max()
        dias_adiante = (data_escolhida - ultima_data).days

        if dias_adiante < 1:
            mensagem = "Por favor, escolha uma data no futuro."
            return render_template('index.html', mensagem=mensagem)

        # Criar datas futuras e fazer a previsão com Prophet
        futuro = modelo.make_future_dataframe(periods=dias_adiante, freq='D')
        previsao = modelo.predict(futuro)

        # Obter o valor previsto para a data selecionada
        previsao_data_escolhida = previsao[previsao['ds'] == data_escolhida]
        if previsao_data_escolhida.empty:
            mensagem = "Não foi possível fazer a previsão para essa data."
            return render_template('index.html', mensagem=mensagem)

        valor_previsto = max(0, int(previsao_data_escolhida['yhat'].values[0]))

        # Retornar o valor previsto ao usuário
        return render_template('index.html', previsao=valor_previsto, data=data_escolhida.strftime('%d/%m/%Y'))

    except Exception as e:
        mensagem = f"Ocorreu um erro: {e}"
        return render_template('index.html', mensagem=mensagem)

# Executar o app
if __name__ == '__main__':
    app.run(debug=True)
