from flask import Flask, request, render_template
import pickle  # Ou 'joblib' se estiver usando joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo recriado
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Coletar os dados enviados pelo formulário
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Fazer a previsão
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text=f'Previsão: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
