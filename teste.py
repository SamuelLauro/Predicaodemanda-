import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# URL da planilha no formato CSV
url = 'https://docs.google.com/spreadsheets/d/17b1N7zApTvUgwx5MaSeD3pHIrGFl9dcIEl3kjNTRUcI/export?format=csv&id=17b1N7zApTvUgwx5MaSeD3pHIrGFl9dcIEl3kjNTRUcI&gid=1297785001'

# Ler os dados da planilha
df = pd.read_csv(url)

# Renomear as colunas removendo espaços extras
df.columns = [col.strip().lower() for col in df.columns]

# Limpar a coluna de números de telefone
df['digite seu número de telefone para contato.'] = df['digite seu número de telefone para contato.'].astype(str).str.replace(r'\D', '', regex=True)

# Mapeamento da frequência de visitas
freq_map = {
    'Semanal': 7,
    'Quinzenal': 14,
    'Mensal': 30
}

# Criar a coluna 'frequência' com base no mapeamento
df['frequência'] = df['com que frequência você visita a barbearia?'].map(freq_map)

# Verificar e tratar valores nulos
df = df.dropna(subset=['frequência'])

# Adicionar características temporais
df.loc[:, 'data'] = pd.to_datetime(df['carimbo de data/hora'], dayfirst=True)
df.loc[:, 'ano'] = df['data'].dt.year
df.loc[:, 'mês'] = df['data'].dt.month
df.loc[:, 'dia'] = df['data'].dt.day
df.loc[:, 'dia_da_semana'] = df['data'].dt.dayofweek

# Selecionar características e variável alvo
X = df[['ano', 'mês', 'dia', 'dia_da_semana']]
y = df['frequência']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Criar e ajustar o pipeline para Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['ano', 'mês', 'dia', 'dia_da_semana'])
        ])),
    ('model', RandomForestRegressor(random_state=0))
])

param_grid_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Melhor modelo Random Forest encontrado
best_model_rf = grid_search_rf.best_estimator_

# Fazer previsões com o melhor modelo Random Forest
y_pred_rf = best_model_rf.predict(X_test)

# Avaliar o modelo Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - Mean Squared Error: {mse_rf}')
print(f'Random Forest - R-squared: {r2_rf}')

# Validação Cruzada para Random Forest
cv_scores_rf = cross_val_score(best_model_rf, X, y, cv=5, scoring='neg_mean_squared_error')

print(f'CV Mean Squared Error (Random Forest): {-cv_scores_rf.mean()}')

# Visualização da frequência de visitas
visit_frequency = df['com que frequência você visita a barbearia?'].value_counts()

# Criar gráfico de barras
plt.figure(figsize=(10, 6))
visit_frequency.plot(kind='bar', color='steelblue', edgecolor='black')

# Adicionar rótulos e título
plt.xlabel('Frequência de Visita', fontsize=12)
plt.ylabel('Número de Clientes', fontsize=12)
plt.title('Número de Clientes por Frequência de Visita', fontsize=14, fontweight='bold')

# Adicionar valores acima das barras
for index, value in enumerate(visit_frequency):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

# Ajustar rotação do eixo X
plt.xticks(rotation=30)

# Adicionar grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()

# Gráfico de pizza
plt.figure(figsize=(8, 8))
visit_frequency.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Distribuição de Frequência de Visitas', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model_rf, f)

print("Modelo salvo como model.pkl")

