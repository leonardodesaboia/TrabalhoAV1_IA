import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

data = np.loadtxt("trabalho/aerogerador.dat", delimiter=None)

X = data[:, 0].reshape(-1, 1)  # N x 1 matriz (coluna)
Y = data[:, 1].reshape(-1, 1)  # N x 1 vetor (coluna)

N,p = X.shape

#Dados com a coluna de 1s (X)
X = np.hstack((
    np.ones((N,1)),X
))


# Organizar as variáveis regressoras (X) e a variável dependente (y)
X = data[:, 0].reshape(-1, 1)  # N x 1 matriz (coluna)
Y = data[:, 1].reshape(-1, 1)  # N x 1 vetor (coluna)

N,p = X.shape

#Dados com a coluna de 1s (X)
X = np.hstack((
    np.ones((N,1)),X
))

#EMBARALHAR O CONJUNTO DE DADOS
idx = np.random.permutation(N)
Xr = X[idx,:]
Yr = Y[idx,:]

#Particionamento do conjunto de dados (80/20)
X_treino = Xr[:int(N*0.8),:]
Y_treino = Yr[:int(N*0.8),:]

X_teste = Xr[int(N*0.8):,:]
Y_teste = Yr[int(N*0.8):,:]

# #Modelo baseado na média:
beta_hat_media = np.array([
        [np.mean(Y_treino)],
        [0],
    ])

#Modelo MQO tradicional
beta_hat = pinv(X_treino.T@X_treino)@X_treino.T@Y_treino
beta_hat_array = np.array(beta_hat)

# #Modelo MQO regularizado (Ridge Regression)
lambda_reg = [0.25, 0.5, 0.75, 1]  # Parâmetro de regularização
I = np.eye(X_treino.shape[1])  # Matriz identidade
beta_hat_ridge = pinv(X_treino.T@X_treino + lambda_reg[0]*I)@X_treino.T@Y_treino

for i in range(len(lambda_reg)):
    beta_hat_ridge = pinv(X_treino.T@X_treino + lambda_reg[i]*I)@X_treino.T@Y_treino
    beta_hat_ridge_array = np.array(beta_hat_ridge)
    print(f"Matriz de betas para lambda = {lambda_reg[i]}:{beta_hat_ridge}")


# #2250 x 2

# # 1. Visualização inicial dos dados - Gráfico de espalhamento
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 1], Y, alpha=0.6, color='blue', s=10)
# plt.xlabel("Velocidade do vento (m/s)")
# plt.ylabel("Potência gerada (kW)")
# plt.title("Gráfico de Espalhamento: Potência vs Velocidade do Vento")
# plt.grid(True, alpha=0.3)
# plt.show()

# print(f"Dimensões dos dados:")
# print(f"- Total de observações (N): {N}")
# print(f"- Variáveis regressoras (p): {p}")
# print(f"- Matriz X: {X.shape} (incluindo coluna de 1s para intercepto)")
# print(f"- Vetor Y: {Y.shape}")
# print(f"- Conjunto de treino: {X_treino.shape[0]} amostras")
# print(f"- Conjunto de teste: {X_teste.shape[0]} amostras")

# print(f"---------------------------------------------------------")

# print(f"Matriz de betas para o modelo baseado na média:")
# print(f"{beta_hat_media}")

# print(f"---------------------------------------------------------") 

# print(f"Matriz de betas para o modelo MQO normal:")
# print(f"{beta_hat}")

# print(f"---------------------------------------------------------") 

# print(f"Matriz de betas para o modelo MQO regularizado:")
# for i in range(len(lambda_reg)):
#     beta_hat_ridge = pinv(X_treino.T@X_treino + lambda_reg[i]*I)@X_treino.T@Y_treino
#     beta_hat_ridge_array = np.array(beta_hat_ridge)
#     print(f"Matriz de betas para lambda = {lambda_reg[i]}:{beta_hat_ridge}")