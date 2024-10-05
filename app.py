from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter a data e converter para o ano
    data = request.form['data']
    ano = int(data.split('-')[0])  # Pega o ano da data no formato YYYY-MM-DD
    
    # Obter outras características
    num_clientes = int(request.form['num_clientes'])
    promocao = 1 if request.form['promocao'] == 'sim' else 0  # Mapeia "sim" para 1 e "não" para 0
    
    # Criar uma lista de características
    int_features = [ano, num_clientes, promocao]
    final_features = np.array(int_features).reshape(1, -1)
    
    # Simular um resultado de previsão
    prediction = np.random.randint(1, 30)  # Simula uma frequência entre 1 e 30
    
    return render_template('index.html', prediction_text=f'Previsão: {prediction} clientes para o dia {data}')

if __name__ == "__main__":
    app.run(debug=True)
