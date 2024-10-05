import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Carregar o modelo salvo
with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter os dados do formulário enviado pelo usuário
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Fazer a previsão
    prediction = modelo.predict(final_features)
    
    return render_template('index.html', prediction_text=f'Previsão de Frequência: {prediction[0]} dias')

if __name__ == "__main__":
    app.run(debug=True)
