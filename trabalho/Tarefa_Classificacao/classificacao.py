import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CLASSES DE CLASSIFICAÇÃO - ESTILO DO PROFESSOR (SIMPLIFICADO)
# ============================================================================

class GaussianClassifier:
    def __init__(self, X_train, y_train, covariance_type='traditional', lambda_reg=0):
        # Organização dos dados
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        self.covariance_type = covariance_type
        self.lambda_reg = lambda_reg
        
        # Separar dados por classe
        self.X = [X_train[:, y_train[0, :] == i] for i in self.classes]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        # Inicializar parâmetros
        self.mu = [None] * self.C
        self.Sigma = [None] * self.C
        self.Sigma_det = [None] * self.C
        self.Sigma_inv = [None] * self.C
        self.P = [None] * self.C

    def fit(self):
        # Calcular parâmetros para cada classe
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
            
            # Calcular matriz de covariância baseada no tipo
            if self.covariance_type == 'traditional':
                self.Sigma[i] = np.cov(self.X[i])
            elif self.covariance_type == 'global':
                # Usar covariância global (reconstruir X_train)
                X_train_full = np.hstack(self.X)
                self.Sigma[i] = np.cov(X_train_full)
            elif self.covariance_type == 'pooled':
                # Covariância agregada
                pooled_cov = np.zeros((self.p, self.p))
                for j in range(self.C):
                    class_cov = np.cov(self.X[j])
                    pooled_cov += (self.n[j] - 1) * class_cov
                self.Sigma[i] = pooled_cov / (self.N - self.C)
            elif self.covariance_type == 'naive':
                # Covariância diagonal (Naive Bayes)
                cov_matrix = np.cov(self.X[i])
                self.Sigma[i] = np.diag(np.diag(cov_matrix))
            elif self.covariance_type == 'friedman':
                # Regularização de Friedman
                sigma = np.cov(self.X[i])
                trace_sigma = np.trace(sigma)
                identity = np.eye(self.p)
                self.Sigma[i] = (1 - self.lambda_reg) * sigma + self.lambda_reg * (trace_sigma / self.p) * identity
            
            # Aplicar lambda baixo preventivamente
            self.Sigma[i] += 1e-6 * np.eye(self.p)
            
            # Calcular determinante e inversa
            self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
            self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])

    def predict(self, x_test):
        # Predição 
        posteriori = [None] * self.C
        for i in range(self.C):
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]
            posteriori[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis
        return self.classes[np.argmax(posteriori)]


class MQOClassifier:
    def __init__(self, X_train, y_train):
        # Organização dos dados 
        self.X_train = X_train
        self.y_train = y_train
        self.W = None
        self.classes = np.arange(1, y_train.shape[1] + 1)
        
    def fit(self):
        # Adicionar coluna de 1s para bias
        X_with_bias = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        
        # MQO com lambda baixo
        XtX = X_with_bias.T @ X_with_bias
        XtX_regularized = XtX + 1e-6 * np.eye(XtX.shape[0])
        self.W = np.linalg.inv(XtX_regularized) @ X_with_bias.T @ self.y_train
        
    def predict(self, x_test):
        # Predição 
        x_with_bias = np.hstack([1, x_test.flatten()])
        scores = x_with_bias @ self.W
        return self.classes[np.argmax(scores)]


# ============================================================================
# FUNÇÕES UTILITÁRIAS
# ============================================================================

def one_hot_encode(y, num_classes):
    """Converte labels para one-hot encoding"""
    valid_mask = (y >= 1) & (y <= num_classes)
    y_filtered = y[valid_mask]
    
    if len(y_filtered) != len(y):
        print(f"Atenção: {len(y) - len(y_filtered)} amostras com labels inválidos foram removidas")
    
    y_one_hot = np.zeros((len(y_filtered), num_classes))
    for i, label in enumerate(y_filtered):
        y_one_hot[i, int(label) - 1] = 1
    return y_one_hot, valid_mask


def accuracy_score(y_true, y_pred):
    """Calcula acurácia"""
    return np.mean(y_true == y_pred)


def run_monte_carlo_validation(X_gauss, y_gauss, X_mqo, y_mqo_onehot, y_data, N, rodadas=50):
    """Executa validação Monte Carlo"""
    
    acuracia_resultados = {
        'mqo_tradicional': [],
        'gauss_tradicional': [],
        'gauss_global': [],
        'gauss_pooled': [],
        'naive_bayes': [],
        'gauss_friedman_025': [],
        'gauss_friedman_050': [],
        'gauss_friedman_075': []
    }
    
    print("Executando simulações...")
    print("Progresso: ", end="")
    
    for r in range(rodadas):
        if r % 5 == 0:
            print(f"{r}", end=" ", flush=True)
        
        # Embaralhar dados
        idx = np.random.permutation(N)
        split_idx = int(N * 0.8)
        train_idx = idx[:split_idx]
        test_idx = idx[split_idx:]
        
        # Dados de treino e teste
        X_train_g = X_gauss[:, train_idx]
        y_train_g = y_gauss[:, train_idx]
        X_test_g = X_gauss[:, test_idx]
        y_test_g = y_gauss[:, test_idx]
        
        X_train_m = X_mqo[train_idx, :]
        y_train_m = y_mqo_onehot[train_idx, :]
        X_test_m = X_mqo[test_idx, :]
        y_test_m = y_data[test_idx]
        
        # 1. MQO Tradicional
        mqo_clf = MQOClassifier(X_train_m, y_train_m)
        mqo_clf.fit()
        
        mqo_predictions = []
        for i in range(len(X_test_m)):
            pred = mqo_clf.predict(X_test_m[i, :])
            mqo_predictions.append(pred)
        
        mqo_acc = accuracy_score(y_test_m, np.array(mqo_predictions))
        acuracia_resultados['mqo_tradicional'].append(mqo_acc)
        
        # 2. Classificador Gaussiano Tradicional
        gauss_trad = GaussianClassifier(X_train_g, y_train_g, 'traditional')
        gauss_trad.fit()
        
        gauss_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_trad.predict(x_test)
            gauss_predictions.append(pred)
        
        gauss_acc = accuracy_score(y_test_g[0, :], np.array(gauss_predictions))
        acuracia_resultados['gauss_tradicional'].append(gauss_acc)
        
        # 3. Classificador Gaussiano Global
        gauss_global = GaussianClassifier(X_train_g, y_train_g, 'global')
        gauss_global.fit()
        
        gauss_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_global.predict(x_test)
            gauss_predictions.append(pred)
        
        gauss_acc = accuracy_score(y_test_g[0, :], np.array(gauss_predictions))
        acuracia_resultados['gauss_global'].append(gauss_acc)
        
        # 4. Classificador Gaussiano Pooled
        gauss_pooled = GaussianClassifier(X_train_g, y_train_g, 'pooled')
        gauss_pooled.fit()
        
        gauss_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_pooled.predict(x_test)
            gauss_predictions.append(pred)
        
        gauss_acc = accuracy_score(y_test_g[0, :], np.array(gauss_predictions))
        acuracia_resultados['gauss_pooled'].append(gauss_acc)
        
        # 5. Naive Bayes
        naive_bayes = GaussianClassifier(X_train_g, y_train_g, 'naive')
        naive_bayes.fit()
        
        naive_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = naive_bayes.predict(x_test)
            naive_predictions.append(pred)
        
        naive_acc = accuracy_score(y_test_g[0, :], np.array(naive_predictions))
        acuracia_resultados['naive_bayes'].append(naive_acc)
        
        # 6-8. Friedman com diferentes lambdas
        for lambda_val, key in [(0.25, 'gauss_friedman_025'), (0.5, 'gauss_friedman_050'), (0.75, 'gauss_friedman_075')]:
            gauss_friedman = GaussianClassifier(X_train_g, y_train_g, 'friedman', lambda_val)
            gauss_friedman.fit()
            
            friedman_predictions = []
            for i in range(X_test_g.shape[1]):
                x_test = X_test_g[:, i].reshape(-1, 1)
                pred = gauss_friedman.predict(x_test)
                friedman_predictions.append(pred)
            
            friedman_acc = accuracy_score(y_test_g[0, :], np.array(friedman_predictions))
            acuracia_resultados[key].append(friedman_acc)
    
    print("\n\nSimulações concluídas!")
    return acuracia_resultados


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

# Carregamento dos dados
data = np.loadtxt("trabalho/EMGsDataset.csv", delimiter=',')

# Organizar dados
if data.shape[0] == 3:
    X_data = data[:2, :].T
    y_data = data[2, :]
    N = data.shape[1]
    p = 2
else:
    X_data = data[:, :-1]
    y_data = data[:, -1]
    N, total_cols = data.shape
    p = total_cols - 1

C = 5

# Filtrar dados válidos
valid_labels_mask = (y_data >= 1) & (y_data <= C)
X_data = X_data[valid_labels_mask]
y_data = y_data[valid_labels_mask]
N = len(y_data)

# Organizar para diferentes modelos
X_gauss = X_data.T
y_gauss = y_data.reshape(1, -1)
X_mqo = X_data
y_mqo_onehot, _ = one_hot_encode(y_data, C)

# Executar validação
acuracia_resultados = run_monte_carlo_validation(X_gauss, y_gauss, X_mqo, y_mqo_onehot, y_data, N, rodadas=500)

# Calcular estatísticas
unique_classes, counts = np.unique(y_data, return_counts=True)
chaves_modelos = [
    'mqo_tradicional', 'gauss_tradicional', 'gauss_global', 'gauss_pooled',
    'naive_bayes', 'gauss_friedman_025', 'gauss_friedman_050', 'gauss_friedman_075'
]

estatisticas = {}
for chave in chaves_modelos:
    acc_valores = np.array(acuracia_resultados[chave])
    estatisticas[chave] = {
        'media': np.mean(acc_valores),
        'desvio': np.std(acc_valores),
        'maximo': np.max(acc_valores),
        'minimo': np.min(acc_valores)
    }


# ============================================================================
# ANÁLISE E PRINTS DOS RESULTADOS
# ============================================================================

print("=" * 80)
print("ANÁLISE DE CLASSIFICAÇÃO EMG - SINAIS FACIAIS")
print("=" * 80)

print("\n1. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("-" * 50)

print(f"Dimensões originais do dataset: {data.shape}")
print(f"Dimensões do dataset: {N} observações, {p} features")
print(f"Número de classes: {C}")
print(f"Classes: 1-Neutro, 2-Sorriso, 3-Sobrancelhas, 4-Surpreso, 5-Rabugento")

print(f"\nDistribuição das classes:")
for cls, count in zip(unique_classes, counts):
    print(f"Classe {int(cls)}: {count} amostras ({count/N*100:.1f}%)")

print(f"\nEstatísticas das features:")
for i in range(p):
    print(f"Feature {i+1}: média={np.mean(X_data[:, i]):.2f}, std={np.std(X_data[:, i]):.2f}")

print("""
ANÁLISE DE SEPARABILIDADE:
1. DISTRIBUIÇÃO ESPACIAL: As classes mostram sobreposições consideráveis
2. SEPARABILIDADE LINEAR: Dados não são linearmente separáveis
3. COMPLEXIDADE: Classificadores não-lineares podem ser mais eficazes
4. RUÍDO: Sinais EMG apresentam variabilidade natural
5. CORRELAÇÃO: Features podem ter dependências que Naive Bayes ignora
""")

print(f"\n2. ORGANIZAÇÃO DOS DADOS")
print("-" * 50)
print(f"Organização para Gaussianos: X_gauss: {X_gauss.shape}, y_gauss: {y_gauss.shape}")
print(f"Organização para MQO: X_mqo: {X_mqo.shape}, y_mqo_onehot: {y_mqo_onehot.shape}")

print(f"\n3. RESULTADOS FINAIS - ESTATÍSTICAS DAS ACURÁCIAS")
print("=" * 80)

nomes_modelos = [
    'MQO tradicional',
    'Classificador Gaussiano Tradicional',
    'Classificador Gaussiano (Cov. de todo cj. treino)',
    'Classificador Gaussiano (Cov. Agregada)',
    'Classificador de Bayes Ingênuo (Naive Bayes)',
    'Classificador Gaussiano Regularizado (Friedman λ=0,25)',
    'Classificador Gaussiano Regularizado (Friedman λ=0,5)',
    'Classificador Gaussiano Regularizado (Friedman λ=0,75)'
]

print(f"{'Modelos':<50} | {'Média':<10} | {'Desvio-Padrão':<13} | {'Maior Valor':<12} | {'Menor Valor':<12}")
print("-" * 50 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 13)

for i, (nome, chave) in enumerate(zip(nomes_modelos, chaves_modelos)):
    stats = estatisticas[chave]
    print(f"{nome:<50} | {stats['media']:<10.4f} | {stats['desvio']:<13.4f} | {stats['maximo']:<12.4f} | {stats['minimo']:<12.4f}")

print("=" * 80)

print(f"\n4. ANÁLISE DOS RESULTADOS")
print("-" * 50)

melhor_modelo = max(estatisticas.keys(), key=lambda k: estatisticas[k]['media'])
idx_melhor = chaves_modelos.index(melhor_modelo)
nome_melhor = nomes_modelos[idx_melhor]

print(f"MELHOR MODELO: {nome_melhor}")
print(f"Acurácia média: {estatisticas[melhor_modelo]['media']:.4f}")
print(f"Desvio padrão: {estatisticas[melhor_modelo]['desvio']:.4f}")

print(f"\nRANKING DOS MODELOS (por acurácia média):")
ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media'], reverse=True)
for i, (chave, stats) in enumerate(ranking):
    idx = chaves_modelos.index(chave)
    nome = nomes_modelos[idx]
    print(f"{i+1}. {nome}: {stats['media']:.4f}")


# ============================================================================
# VISUALIZAÇÃO E GRÁFICOS
# ============================================================================

print(f"\n5. GRÁFICOS COMPARATIVOS")
print("-" * 50)

# Figura 1: Gráfico de espalhamento
plt.figure(1, figsize=(12, 8))

cores = ['red', 'blue', 'green', 'orange', 'purple']
nomes_classes = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']

for i, (classe, cor, nome) in enumerate(zip(unique_classes, cores, nomes_classes)):
    mask = y_data == classe
    plt.scatter(X_data[mask, 0], X_data[mask, 1], 
               c=cor, label=f'{int(classe)}-{nome}', alpha=0.6, s=20)

plt.xlabel('Sensor 1 - Corrugador do Supercílio')
plt.ylabel('Sensor 2 - Zigomático Maior') 
plt.title('Gráfico de Espalhamento - Sinais EMG por Expressão Facial')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Figura 2: Análises comparativas
plt.figure(2, figsize=(15, 10))

nomes_curtos = ['MQO', 'G.Tradicional', 'G.cj_treino', 'G.Agregada', 'Naive-Bayes', 'Friedman-0.25', 'Friedman-0.5', 'Friedman-0.75']
cores_box = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

# Gráfico 1: Boxplot
plt.subplot(2, 2, 1)
dados_boxplot = [acuracia_resultados[chave] for chave in chaves_modelos]
box_plot = plt.boxplot(dados_boxplot, labels=nomes_curtos, patch_artist=True)
for patch, cor in zip(box_plot['boxes'], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
plt.title('Distribuição das Acurácias por Modelo')
plt.ylabel('Acurácia')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 2: Médias
plt.subplot(2, 2, 2)
medias_acc = [estatisticas[chave]['media'] for chave in chaves_modelos]
bars = plt.bar(nomes_curtos, medias_acc, color=cores_box, alpha=0.8)
plt.title('Acurácia Média por Modelo')
plt.ylabel('Acurácia Média')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for bar, valor in zip(bars, medias_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{valor:.3f}', ha='center', va='bottom', fontsize=9)

# Gráfico 3: Desvios
plt.subplot(2, 2, 3)
desvios_acc = [estatisticas[chave]['desvio'] for chave in chaves_modelos]
bars = plt.bar(nomes_curtos, desvios_acc, color=cores_box, alpha=0.8)
plt.title('Desvio Padrão das Acurácias')
plt.ylabel('Desvio Padrão')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for bar, valor in zip(bars, desvios_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{valor:.3f}', ha='center', va='bottom', fontsize=9)

# Gráfico 4: Efeito Friedman
plt.subplot(2, 2, 4)
lambdas = [0, 0.25, 0.5, 0.75]
modelos_friedman = ['gauss_tradicional', 'gauss_friedman_025', 'gauss_friedman_050', 'gauss_friedman_075']
acc_friedman = [estatisticas[modelo]['media'] for modelo in modelos_friedman]

plt.plot(lambdas, acc_friedman, 'bo-', linewidth=2, markersize=8)
plt.title('Efeito da Regularização Friedman')
plt.xlabel('Parâmetro λ')
plt.ylabel('Acurácia Média')
plt.grid(True, alpha=0.3)

for x, y in zip(lambdas, acc_friedman):
    plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\nAnálise de classificação concluída!")
print("Figure 1: Gráfico de espalhamento dos dados EMG")
print("Figure 2: Análises comparativas dos classificadores")
print(f"\nLEGENDA DOS MODELOS:")
print("MQO = MQO tradicional")
print("G.Tradicional = Classificador Gaussiano Tradicional")
print("G.cj_treino = Classificador Gaussiano (Cov. de todo conjunto treino)")
print("G.Agregada = Classificador Gaussiano (Cov. Agregada)")
print("Naive-Bayes = Classificador de Bayes Ingênuo")
print("Friedman-0.25/0.5/0.75 = Classificador Gaussiano Regularizado")
print("=" * 80)