import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Carregar dados do EMG
data = np.loadtxt("trabalho/EMGsDataset.csv", delimiter=',')

# Formato original: 3 linhas x 50000 colunas
# Linha 0: Sensor 1 (Corrugador do Supercílio)
# Linha 1: Sensor 2 (Zigomático Maior)  
# Linha 2: Labels (1-5)

print("TAREFA DE CLASSIFICAÇÃO - SINAIS EMG FACIAIS")
print("=" * 60)
print(f"Formato original dos dados: {data.shape}")
print(f"N = {data.shape[1]} amostras")
print(f"p = 2 características (sensores)")
print(f"C = 5 classes (expressões faciais)")

# ====================================================================
# 1. ORGANIZAÇÃO DOS DADOS (QUESTÃO 1)
# ====================================================================

# Extrair características e labels
sensor1 = data[0, :]  # Corrugador do Supercílio
sensor2 = data[1, :]  # Zigomático Maior
labels = data[2, :].astype(int)  # Classes 1-5

N = data.shape[1]  # 50000 amostras
p = 2  # 2 características
C = 5  # 5 classes

# Para MQO: X ∈ R^(N×p), Y ∈ R^(N×C)
X_mqo = np.column_stack((sensor1, sensor2))  # N x p

# One-hot encoding para MQO
Y_mqo = np.zeros((N, C))
for i, label in enumerate(labels):
    Y_mqo[i, label-1] = 1  # label-1 pois classes são 1-5

# Para Gaussianos: X ∈ R^(p×N), Y ∈ R^(C×N)  
X_gauss = X_mqo.T  # p x N
Y_gauss = Y_mqo.T  # C x N

print(f"\nOrganização dos dados:")
print(f"Para MQO: X{X_mqo.shape}, Y{Y_mqo.shape}")
print(f"Para Gaussianos: X{X_gauss.shape}, Y{Y_gauss.shape}")

# ====================================================================
# 2. VISUALIZAÇÃO INICIAL DOS DADOS (QUESTÃO 2)
# ====================================================================

plt.figure(figsize=(10, 8))
cores = ['red', 'blue', 'green', 'orange', 'purple']
nomes_classes = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']

for i in range(1, 6):
    mask = labels == i
    plt.scatter(sensor1[mask], sensor2[mask], 
               c=cores[i-1], label=f'{i} - {nomes_classes[i-1]}', 
               alpha=0.6, s=10)

plt.xlabel('Sensor 1 - Corrugador do Supercílio')
plt.ylabel('Sensor 2 - Zigomático Maior')
plt.title('Visualização dos Dados EMG\n(Sinais dos Músculos Faciais)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=False)

# ====================================================================
# 3. IMPLEMENTAÇÃO DOS CLASSIFICADORES (QUESTÃO 3)
# ====================================================================

class MQOClassifier:
    """Classificador baseado em Mínimos Quadrados Ordinários"""
    def __init__(self):
        self.W = None
        
    def fit(self, X, Y):
        # W = (X^T X)^-1 X^T Y
        self.W = pinv(X.T @ X) @ X.T @ Y
        
    def predict(self, X):
        predictions = X @ self.W
        return np.argmax(predictions, axis=1) + 1  # +1 pois classes são 1-5

class GaussianClassifier:
    """Classificador Gaussiano - baseado no código do professor"""
    def __init__(self, X_train, y_train, classifier_type='traditional'):
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        self.classifier_type = classifier_type
        
        # Separar dados por classe
        self.X = [X_train[:, y_train[0, :] == i] for i in self.classes]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        # Inicializar parâmetros
        self.mu = [None] * self.C
        self.Sigma = [None] * self.C
        self.Sigma_det = [None] * self.C
        self.Sigma_inv = [None] * self.C
        self.P = [None] * self.C  # Probabilidades a priori
        
    def fit(self, lambda_reg=0):
        # Calcular médias por classe
        for i in range(self.C):
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
            
        if self.classifier_type == 'traditional':
            # Matriz de covariância individual por classe
            for i in range(self.C):
                self.Sigma[i] = np.cov(self.X[i])
                self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])
                
        elif self.classifier_type == 'pooled':
            # Matriz de covariância agregada (pooled)
            Sigma_pooled = np.zeros((self.p, self.p))
            for i in range(self.C):
                Sigma_pooled += (self.n[i] - 1) * np.cov(self.X[i])
            Sigma_pooled /= (self.N - self.C)
            
            for i in range(self.C):
                self.Sigma[i] = Sigma_pooled
                self.Sigma_det[i] = np.linalg.det(Sigma_pooled)
                self.Sigma_inv[i] = np.linalg.inv(Sigma_pooled)
                
        elif self.classifier_type == 'friedman':
            # RDA - Regularized Discriminant Analysis
            # Primeiro calcula matriz pooled
            Sigma_pooled = np.zeros((self.p, self.p))
            for i in range(self.C):
                Sigma_pooled += (self.n[i] - 1) * np.cov(self.X[i])
            Sigma_pooled /= (self.N - self.C)
            
            # Aplica regularização para cada classe
            for i in range(self.C):
                Sigma_individual = np.cov(self.X[i])
                # Friedman: Sigma = (1-λ) * Sigma_individual + λ * Sigma_pooled
                self.Sigma[i] = (1 - lambda_reg) * Sigma_individual + lambda_reg * Sigma_pooled
                
                # Adicionar regularização diagonal se necessário para estabilidade
                if lambda_reg > 0:
                    self.Sigma[i] += 1e-6 * np.eye(self.p)
                    
                self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])
                
        elif self.classifier_type == 'naive':
            # Naive Bayes - matriz diagonal (assume independência)
            for i in range(self.C):
                Sigma_full = np.cov(self.X[i])
                self.Sigma[i] = np.diag(np.diag(Sigma_full))  # Só diagonal
                self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])
    
    def predict(self, x_test):
        posteriori = [None] * self.C
        for i in range(self.C):
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]
            posteriori[i] = np.log(self.P[i]) - 1/2 * np.log(self.Sigma_det[i]) - 1/2 * d_mahalanobis
        return np.argmax(posteriori) + 1

# ====================================================================
# 4. SIMULAÇÃO MONTE CARLO (QUESTÕES 4 e 5)
# ====================================================================

print("\nIniciando simulação Monte Carlo com 50 rodadas...")
print("=" * 60)

# Listas para armazenar acurácias
acc_mqo = []
acc_gauss_trad = []
acc_gauss_pooled = []
acc_gauss_friedman_0 = []
acc_gauss_friedman_025 = []
acc_gauss_friedman_05 = []
acc_gauss_friedman_075 = []
acc_gauss_friedman_1 = []
acc_naive_bayes = []

# Valores de lambda para Friedman
lambda_values = [0, 0.25, 0.5, 0.75, 1]

# 50 rodadas de Monte Carlo (mais prático que 500)
for rodada in range(50):
    if (rodada + 1) % 10 == 0:
        print(f"Rodada {rodada + 1}/50...")
    
    # Embaralhar dados
    idx = np.random.permutation(N)
    X_mqo_shuffled = X_mqo[idx, :]
    Y_mqo_shuffled = Y_mqo[idx, :]
    X_gauss_shuffled = X_mqo_shuffled.T
    Y_gauss_shuffled = Y_mqo_shuffled.T
    labels_shuffled = labels[idx]
    
    # Split 80/20
    split_idx = int(0.8 * N)
    
    # Dados para MQO
    X_train_mqo = X_mqo_shuffled[:split_idx, :]
    Y_train_mqo = Y_mqo_shuffled[:split_idx, :]
    X_test_mqo = X_mqo_shuffled[split_idx:, :]
    Y_test_mqo = Y_mqo_shuffled[split_idx:, :]
    
    # Dados para Gaussianos
    X_train_gauss = X_gauss_shuffled[:, :split_idx]
    Y_train_gauss = Y_gauss_shuffled[:, :split_idx]
    X_test_gauss = X_gauss_shuffled[:, split_idx:]
    Y_test_gauss = Y_gauss_shuffled[:, split_idx:]
    labels_train = labels_shuffled[:split_idx]
    labels_test = labels_shuffled[split_idx:]
    
    # 1. MQO Tradicional
    try:
        mqo = MQOClassifier()
        mqo.fit(X_train_mqo, Y_train_mqo)
        pred_mqo = mqo.predict(X_test_mqo)
        acc_mqo.append(np.mean(pred_mqo == labels_test))
    except:
        acc_mqo.append(0.0)  # Em caso de erro
    
    # 2. Gaussiano Tradicional
    try:
        gc_trad = GaussianClassifier(X_train_gauss, labels_train.reshape(1, -1), 'traditional')
        gc_trad.fit()
        
        predictions = []
        for j in range(len(labels_test)):
            x_test = X_test_gauss[:, j].reshape(2, 1)
            pred = gc_trad.predict(x_test)
            predictions.append(pred)
        
        acc_gauss_trad.append(np.mean(np.array(predictions) == labels_test))
    except:
        acc_gauss_trad.append(0.0)
    
    # 3. Gaussiano Pooled
    try:
        gc_pooled = GaussianClassifier(X_train_gauss, labels_train.reshape(1, -1), 'pooled')
        gc_pooled.fit()
        
        predictions = []
        for j in range(len(labels_test)):
            x_test = X_test_gauss[:, j].reshape(2, 1)
            pred = gc_pooled.predict(x_test)
            predictions.append(pred)
        
        acc_gauss_pooled.append(np.mean(np.array(predictions) == labels_test))
    except:
        acc_gauss_pooled.append(0.0)
    
    # 4. Gaussiano Friedman (5 valores de lambda)
    acc_friedman_rodada = []
    for lam in lambda_values:
        try:
            gc_friedman = GaussianClassifier(X_train_gauss, labels_train.reshape(1, -1), 'friedman')
            gc_friedman.fit(lam)
            
            predictions = []
            for j in range(len(labels_test)):
                x_test = X_test_gauss[:, j].reshape(2, 1)
                pred = gc_friedman.predict(x_test)
                predictions.append(pred)
            
            acc = np.mean(np.array(predictions) == labels_test)
            acc_friedman_rodada.append(acc)
        except:
            acc_friedman_rodada.append(0.0)
    
    # Armazenar acurácias do Friedman
    acc_gauss_friedman_0.append(acc_friedman_rodada[0])
    acc_gauss_friedman_025.append(acc_friedman_rodada[1])
    acc_gauss_friedman_05.append(acc_friedman_rodada[2])
    acc_gauss_friedman_075.append(acc_friedman_rodada[3])
    acc_gauss_friedman_1.append(acc_friedman_rodada[4])
    
    # 5. Naive Bayes
    try:
        gc_naive = GaussianClassifier(X_train_gauss, labels_train.reshape(1, -1), 'naive')
        gc_naive.fit()
        
        predictions = []
        for j in range(len(labels_test)):
            x_test = X_test_gauss[:, j].reshape(2, 1)
            pred = gc_naive.predict(x_test)
            predictions.append(pred)
        
        acc_naive_bayes.append(np.mean(np.array(predictions) == labels_test))
    except:
        acc_naive_bayes.append(0.0)

print("Simulação Monte Carlo concluída!")
print("=" * 60)

# ====================================================================
# 6. ANÁLISE ESTATÍSTICA DOS RESULTADOS (QUESTÃO 6)
# ====================================================================

# Organizar dados para análise
modelos = ['MQO', 'Gauss_Trad', 'Gauss_Pooled', 'Friedman_λ=0', 
           'Friedman_λ=0.25', 'Friedman_λ=0.5', 'Friedman_λ=0.75', 
           'Friedman_λ=1', 'Naive_Bayes']

acc_listas = [acc_mqo, acc_gauss_trad, acc_gauss_pooled, acc_gauss_friedman_0,
              acc_gauss_friedman_025, acc_gauss_friedman_05, acc_gauss_friedman_075,
              acc_gauss_friedman_1, acc_naive_bayes]

# Calcular estatísticas
estatisticas = []
for i, acc_lista in enumerate(acc_listas):
    stats = {
        'Modelo': modelos[i],
        'Media': np.mean(acc_lista),
        'Desvio_Padrao': np.std(acc_lista),
        'Minimo': np.min(acc_lista),
        'Maximo': np.max(acc_lista),
        'Mediana': np.median(acc_lista)
    }
    estatisticas.append(stats)

# TABELA DE RESULTADOS
print("\n" + "="*90)
print("TABELA DE ESTATÍSTICAS DAS ACURÁCIAS (50 RODADAS MONTE CARLO)")
print("="*90)
print(f"{'Modelo':<15} {'Média':<10} {'Desvio':<10} {'Mínimo':<10} {'Máximo':<10} {'Mediana':<10}")
print("-" * 90)

for stats in estatisticas:
    print(f"{stats['Modelo']:<15} {stats['Media']:<10.4f} {stats['Desvio_Padrao']:<10.4f} "
          f"{stats['Minimo']:<10.4f} {stats['Maximo']:<10.4f} {stats['Mediana']:<10.4f}")

print("="*90)

# RANKING DE PERFORMANCE
ranking = sorted(estatisticas, key=lambda x: x['Media'], reverse=True)
print(f"\nRANKING DE PERFORMANCE (Acurácia média - maior é melhor):")
print("-" * 60)
for i, modelo in enumerate(ranking, 1):
    print(f"{i}º lugar: {modelo['Modelo']} - Acurácia: {modelo['Media']:.4f}")

# GRÁFICOS DE ANÁLISE
print(f"\nGerando gráficos de análise...")

plt.figure(figsize=(15, 12))

# Gráfico 1: Boxplot comparativo
plt.subplot(2, 3, 1)
plt.boxplot(acc_listas, labels=modelos)
plt.title('Distribuição das Acurácias por Modelo\n(50 rodadas Monte Carlo)')
plt.ylabel('Acurácia')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 2: Barras com médias e desvios
plt.subplot(2, 3, 2)
medias = [stats['Media'] for stats in estatisticas]
desvios = [stats['Desvio_Padrao'] for stats in estatisticas]
x_pos = range(len(modelos))

plt.bar(x_pos, medias, yerr=desvios, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Acurácia Média ± Desvio Padrão')
plt.ylabel('Acurácia')
plt.xticks(x_pos, modelos, rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 3: Comparação dos valores de lambda (Friedman)
plt.subplot(2, 3, 3)
lambda_accs = [acc_gauss_friedman_0, acc_gauss_friedman_025, acc_gauss_friedman_05, 
               acc_gauss_friedman_075, acc_gauss_friedman_1]
lambda_labels = ['λ=0', 'λ=0.25', 'λ=0.5', 'λ=0.75', 'λ=1']

plt.boxplot(lambda_accs, labels=lambda_labels)
plt.title('Impacto do λ no Classificador Friedman')
plt.ylabel('Acurácia')
plt.grid(True, alpha=0.3)

# Gráfico 4: Histograma do melhor modelo
melhor_modelo_nome = ranking[0]['Modelo']
melhor_idx = modelos.index(melhor_modelo_nome)
melhor_acc = acc_listas[melhor_idx]

plt.subplot(2, 3, 4)
plt.hist(melhor_acc, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title(f'Distribuição - {melhor_modelo_nome}\n(Melhor modelo)')
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)

# Gráfico 5: Scatter Performance vs Estabilidade
plt.subplot(2, 3, 5)
cores_scatter = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for i, stats in enumerate(estatisticas):
    plt.scatter(stats['Media'], stats['Desvio_Padrao'],
               s=100, c=cores_scatter[i], label=stats['Modelo'][:8], alpha=0.7)

plt.xlabel('Acurácia Média')
plt.ylabel('Desvio Padrão')
plt.title('Performance vs Estabilidade')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Gráfico 6: Evolução do lambda
plt.subplot(2, 3, 6)
lambda_means = [np.mean(acc) for acc in lambda_accs]
plt.plot([0, 0.25, 0.5, 0.75, 1], lambda_means, 'o-', linewidth=2, markersize=8)
plt.title('Evolução da Performance\ncom λ (Friedman)')
plt.xlabel('λ (Lambda)')
plt.ylabel('Acurácia Média')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# DISCUSSÃO DOS RESULTADOS
print("\n" + "="*90)
print("DISCUSSÃO DOS RESULTADOS")
print("="*90)

print(f"\n1. PERFORMANCE GERAL:")
print(f"   • Melhor modelo: {ranking[0]['Modelo']} (Acurácia: {ranking[0]['Media']:.4f})")
print(f"   • Pior modelo: {ranking[-1]['Modelo']} (Acurácia: {ranking[-1]['Media']:.4f})")
print(f"   • Diferença: {ranking[0]['Media'] - ranking[-1]['Media']:.4f}")

print(f"\n2. ESTABILIDADE:")
ranking_estabilidade = sorted(estatisticas, key=lambda x: x['Desvio_Padrao'])
print(f"   • Mais estável: {ranking_estabilidade[0]['Modelo']} (σ: {ranking_estabilidade[0]['Desvio_Padrao']:.4f})")
print(f"   • Menos estável: {ranking_estabilidade[-1]['Modelo']} (σ: {ranking_estabilidade[-1]['Desvio_Padrao']:.4f})")

print(f"\n3. ANÁLISE DO REGULARIZADOR FRIEDMAN:")
melhor_lambda_idx = np.argmax(lambda_means)
melhor_lambda = lambda_values[melhor_lambda_idx]
print(f"   • Melhor λ: {melhor_lambda} (Acurácia: {lambda_means[melhor_lambda_idx]:.4f})")
print(f"   • Impacto da regularização: {'Positivo' if lambda_means[melhor_lambda_idx] > lambda_means[0] else 'Negativo'}")

print(f"\n4. COMPARAÇÃO ENTRE ABORDAGENS:")
print(f"   • MQO vs melhor Gaussiano: {np.mean(acc_mqo):.4f} vs {max(lambda_means + [np.mean(acc_gauss_trad), np.mean(acc_gauss_pooled), np.mean(acc_naive_bayes)]):.4f}")
print(f"   • Naive Bayes efetividade: {np.mean(acc_naive_bayes):.4f}")
print(f"   • Pooled vs Traditional: {np.mean(acc_gauss_pooled):.4f} vs {np.mean(acc_gauss_trad):.4f}")

print(f"\n5. CONCLUSÕES:")
print(f"   • Separabilidade: {'Alta' if ranking[0]['Media'] > 0.8 else 'Moderada' if ranking[0]['Media'] > 0.6 else 'Baixa'}")
print(f"   • Robustez dos gaussianos: Regularização {'ajudou' if max(lambda_means[1:]) > lambda_means[0] else 'prejudicou'}")
print(f"   • Aplicabilidade prática: Modelo recomendado é {ranking[0]['Modelo']}")

print("="*90)