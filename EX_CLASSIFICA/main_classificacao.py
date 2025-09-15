import numpy as np
import matplotlib.pyplot as plt
import os
from gaussian_classifiers import GaussianClassifier

# Corrigir caminho para o arquivo CSV
csv_path = os.path.join(os.path.dirname(__file__), "EMG3Classes.csv")
data = np.loadtxt(csv_path, delimiter=',')

X, y  = data[:,:-1], data[:,-1:]

X_train = X[:int(.8*X.shape[0]),:]
y_train = y[:int(.8*X.shape[0]),:]

x_test = X[int(.8*X.shape[0]),:].reshape(2,1)
y_test = y[int(.8*X.shape[0]),:]

gc = GaussianClassifier(X_train.T,y_train.T)
gc.fit()
y_pred = gc.predict(x_test)

# ====================================================================
# ANÁLISE ESTATÍSTICA DOS RESULTADOS
# ====================================================================

# Mostrar resultados
print(f"Predição: {y_pred}")
print(f"Valor real: {y_test[0]}")
print(f"Correto: {'Sim' if y_pred == y_test[0] else 'Não'}")

# Testar em mais amostras de teste
X_test = X[int(.8*X.shape[0]):,:]
y_test_all = y[int(.8*X.shape[0]):]

predictions = []
for i in range(len(X_test)):
    x_test_i = X_test[i,:].reshape(2,1)
    pred = gc.predict(x_test_i)
    predictions.append(pred)

predictions = np.array(predictions)
accuracy = np.mean(predictions == y_test_all.flatten())
print(f"Acurácia total: {accuracy:.2%}")

# Criar gráficos
plt.figure(figsize=(12, 4))

# Gráfico 1: Dados de treino
plt.subplot(1, 3, 1)
cores = ['red', 'blue', 'green']
for i, classe in enumerate(np.unique(y_train)):
    mask = y_train.flatten() == classe
    plt.scatter(X_train[mask, 0], X_train[mask, 1], 
               c=cores[i], label=f'Classe {int(classe)}', alpha=0.7)
plt.title('Dados de Treino')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Gráfico 2: Dados de teste (valores reais)
plt.subplot(1, 3, 2)
for i, classe in enumerate(np.unique(y_test_all)):
    mask = y_test_all.flatten() == classe
    plt.scatter(X_test[mask, 0], X_test[mask, 1], 
               c=cores[i], label=f'Classe {int(classe)}', alpha=0.7)
plt.title('Teste - Valores Reais')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Gráfico 3: Predições
plt.subplot(1, 3, 3)
for i, classe in enumerate(np.unique(predictions)):
    mask = predictions == classe
    plt.scatter(X_test[mask, 0], X_test[mask, 1], 
               c=cores[i], label=f'Predito {int(classe)}', alpha=0.7)
plt.title('Teste - Predições')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

bp=1