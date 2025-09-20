import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class GaussianClassifier:
    """
    Classificador Bayesiano Gaussiano conforme os slides do professor
    Implementa os métodos: tradicional, covariâncias iguais, agregada, naive bayes e Friedman
    """

    def __init__(self, method='traditional', lambda_reg=0.0):
        """
        Parâmetros:
        method: 'traditional', 'equal_cov', 'aggregated', 'naive_bayes', 'friedman'
        lambda_reg: parâmetro λ para regularização de Friedman
        """
        self.method = method
        self.lambda_reg = lambda_reg
        self.classes = None
        self.C = None
        self.p = None
        self.N = None
        self.mu = None
        self.Sigma = None
        self.Sigma_inv = None
        self.Sigma_det = None
        self.P_prior = None
        
    def fit(self, X_train, y_train):
        """
        Treina o classificador
        X_train: matriz (p × N) - conforme slides
        y_train: vetor (1 × N) - conforme slides
        """
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        X_by_class = {}
        n_by_class = {}
        
        for cls in self.classes:
            mask = y_train[0, :] == cls
            X_by_class[cls] = X_train[:, mask]
            n_by_class[cls] = np.sum(mask)
        
        self.mu = {}
        for cls in self.classes:
            self.mu[cls] = np.mean(X_by_class[cls], axis=1).reshape(self.p, 1)
        
        # Calcular probabilidades a priori (slide 33)
        self.P_prior = {}
        for cls in self.classes:
            self.P_prior[cls] = n_by_class[cls] / self.N
        
        if self.method == 'traditional':
            self._compute_traditional_covariance(X_by_class)
        elif self.method == 'equal_cov':
            self._compute_equal_covariance(X_train)
        elif self.method == 'aggregated':
            self._compute_aggregated_covariance(X_by_class, n_by_class)
        elif self.method == 'naive_bayes':
            self._compute_naive_bayes_covariance(X_by_class)
        elif self.method == 'friedman':
            self._compute_friedman_covariance(X_by_class, n_by_class)
        
        
        self.Sigma_det = {}
        self.Sigma_inv = {}
        
        for cls in self.classes:
            try:
                self.Sigma_det[cls] = np.linalg.det(self.Sigma[cls])
                self.Sigma_inv[cls] = np.linalg.inv(self.Sigma[cls])
            except np.linalg.LinAlgError:            
                reg_matrix = self.Sigma[cls] + 1e-6 * np.eye(self.p)
                self.Sigma_det[cls] = np.linalg.det(reg_matrix)
                self.Sigma_inv[cls] = np.linalg.inv(reg_matrix)
    
    def _compute_traditional_covariance(self, X_by_class):
        """Algoritmo 1 - Matriz de covariância para cada classe"""
        self.Sigma = {}
        for cls in self.classes:
            self.Sigma[cls] = np.cov(X_by_class[cls])
    
    def _compute_equal_covariance(self, X_train):
        """Slide 50 - Covariâncias iguais (LDA)"""

        Sigma_common = np.cov(X_train)
        self.Sigma = {}
        for cls in self.classes:
            self.Sigma[cls] = Sigma_common
    
    def _compute_aggregated_covariance(self, X_by_class, n_by_class):
        """Slide 52 - Matriz agregada ponderada"""
        Sigma_aggregated = np.zeros((self.p, self.p))
        
        for cls in self.classes:
            weight = n_by_class[cls] / self.N  # P(yi)
            Sigma_i = np.cov(X_by_class[cls])
            Sigma_aggregated += weight * Sigma_i
        
        self.Sigma = {}
        for cls in self.classes:
            self.Sigma[cls] = Sigma_aggregated
    
    def _compute_naive_bayes_covariance(self, X_by_class):
        """Slide 57 - Matriz diagonal (atributos independentes)"""
        self.Sigma = {}
        for cls in self.classes:
            # Calcular covariância completa e manter apenas diagonal
            full_cov = np.cov(X_by_class[cls])
            self.Sigma[cls] = np.diag(np.diag(full_cov))
    
    def _compute_friedman_covariance(self, X_by_class, n_by_class):
        """Slide 54 - Regularização de Friedman"""

        Sigma_aggregated = np.zeros((self.p, self.p))
        for cls in self.classes:
            weight = n_by_class[cls] / self.N
            Sigma_i = np.cov(X_by_class[cls])
            Sigma_aggregated += weight * Sigma_i
        
        
        self.Sigma = {}
        for cls in self.classes:
            ni = n_by_class[cls]
            Sigma_i = np.cov(X_by_class[cls])
            
            numerator = (1 - self.lambda_reg) * ni * Sigma_i + self.lambda_reg * self.N * Sigma_aggregated
            denominator = (1 - self.lambda_reg) * ni + self.lambda_reg * self.N
            
            self.Sigma[cls] = numerator / denominator
    
    def predict(self, x_test):
        """
        Predição usando critério MAP (slides 43-47)
        x_test: vetor coluna (p × 1)
        """
        posteriors = {}
        
        for cls in self.classes:
            diff = x_test - self.mu[cls]
            mahalanobis_dist = (diff.T @ self.Sigma_inv[cls] @ diff)[0, 0]
            
            # Função discriminante (slide 45)
            if self.method == 'equal_cov' or self.method == 'aggregated':
                posteriors[cls] = np.log(self.P_prior[cls]) - 0.5 * mahalanobis_dist
            else:
                posteriors[cls] = (np.log(self.P_prior[cls]) - 
                                 0.5 * np.log(self.Sigma_det[cls]) - 
                                 0.5 * mahalanobis_dist)
        
        return max(posteriors, key=posteriors.get)


class MQOClassifier:
    """
    Classificador por Mínimos Quadrados Ordinários
    """
    
    def __init__(self):
        self.W = None
        self.classes = None
    
    def fit(self, X_train, y_train):
        """
        X_train: (N × p) para MQO
        y_train: (N × C) one-hot encoded
        """
        # Adicionar intercepto
        X_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        
        # MQO
        XtX = X_with_bias.T @ X_with_bias
        XtY = X_with_bias.T @ y_train
        
        try:
            self.W = np.linalg.inv(XtX) @ XtY
        except np.linalg.LinAlgError:
            self.W = np.linalg.pinv(XtX) @ XtY
        
        self.classes = np.arange(1, y_train.shape[1] + 1)
    
    def predict(self, x_test):
        """Predição para uma amostra"""
        x_with_bias = np.hstack([1, x_test.flatten()])
        
        scores = x_with_bias @ self.W
        
        return self.classes[np.argmax(scores)]


def one_hot_encode(y, num_classes):
    """Converte labels para one-hot encoding"""
    y_onehot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        if 1 <= label <= num_classes:
            y_onehot[i, int(label) - 1] = 1
    return y_onehot


def accuracy_score(y_true, y_pred):
    """Calcula acurácia"""
    return np.mean(y_true == y_pred)


def monte_carlo_validation(X_gauss, y_gauss, X_mqo, y_mqo_onehot, y_labels, num_runs=10):
    """
    Validação Monte Carlo conforme especificado no trabalho
    """
    
    models_config = [
        ('MQO tradicional', 'mqo'),
        ('Classificador Gaussiano Tradicional', 'traditional'),
        ('Classificador Gaussiano (Cov. de todo cj. treino)', 'equal_cov'),
        ('Classificador Gaussiano (Cov. Agregada)', 'aggregated'),
        ('Classificador de Bayes Ingênuo', 'naive_bayes'),
        ('Classificador Gaussiano Regularizado (Friedman lambda=0.25)', 'friedman_025'),
        ('Classificador Gaussiano Regularizado (Friedman lambda=0.5)', 'friedman_050'),
        ('Classificador Gaussiano Regularizado (Friedman lambda=0.75)', 'friedman_075')
    ]
    
    results = {name: [] for name, _ in models_config}
    
    N = X_gauss.shape[1]
    
    print(f"Executando {num_runs} simulações Monte Carlo...")
    print("Progresso: ", end="", flush=True)
    
    for run in range(num_runs):
        if run % 5 == 0:
            print(f"{run} ", end="", flush=True)
        
        indices = np.random.permutation(N)
        split_idx = int(0.8 * N)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        # Dados de treino e teste
        X_train_g = X_gauss[:, train_idx]
        y_train_g = y_gauss[:, train_idx]
        X_test_g = X_gauss[:, test_idx]
        y_test_g = y_gauss[:, test_idx]
        
        X_train_m = X_mqo[train_idx, :]
        y_train_m = y_mqo_onehot[train_idx, :]
        X_test_m = X_mqo[test_idx, :]
        y_test_labels = y_labels[test_idx]
        
        for model_name, model_type in models_config:
            try:
                if model_type == 'mqo':
                    # MQO
                    clf = MQOClassifier()
                    clf.fit(X_train_m, y_train_m)
                    
                    predictions = []
                    for i in range(len(X_test_m)):
                        pred = clf.predict(X_test_m[i, :])
                        predictions.append(pred)
                    
                    acc = accuracy_score(y_test_labels, np.array(predictions))
                    
                elif model_type.startswith('friedman'):
                    # Friedman
                    lambda_val = float(model_type.split('_')[1]) / 100
                    clf = GaussianClassifier(method='friedman', lambda_reg=lambda_val)
                    clf.fit(X_train_g, y_train_g)
                    
                    predictions = []
                    for i in range(X_test_g.shape[1]):
                        x_test = X_test_g[:, i].reshape(-1, 1)
                        pred = clf.predict(x_test)
                        predictions.append(pred)
                    
                    acc = accuracy_score(y_test_g[0, :], np.array(predictions))
                    
                else:
                    clf = GaussianClassifier(method=model_type)
                    clf.fit(X_train_g, y_train_g)
                    
                    predictions = []
                    for i in range(X_test_g.shape[1]):
                        x_test = X_test_g[:, i].reshape(-1, 1)
                        pred = clf.predict(x_test)
                        predictions.append(pred)
                    
                    acc = accuracy_score(y_test_g[0, :], np.array(predictions))
                
                results[model_name].append(acc)
                
            except Exception as e:
                results[model_name].append(0.0)
    
    print(f"\nSimulações concluídas!")
    return results


def print_results_table(results):
    """Imprime tabela de resultados conforme solicitado"""
    
    print("\n" + "="*100)
    print("TABELA DE RESULTADOS - VALIDAÇÃO MONTE CARLO")
    print("="*100)
    
    print(f"{'Modelos':<60} | {'Média':<10} | {'Desvio-Padrão':<13} | {'Maior Valor':<12} | {'Menor Valor':<12}")
    print("-"*60 + "|" + "-"*11 + "|" + "-"*14 + "|" + "-"*13 + "|" + "-"*13)
    
    stats = {}
    for model_name, accuracies in results.items():
        acc_array = np.array(accuracies)
        
        media = np.mean(acc_array)
        desvio = np.std(acc_array)
        maximo = np.max(acc_array)
        minimo = np.min(acc_array)
        
        stats[model_name] = {
            'media': media,
            'desvio': desvio,
            'maximo': maximo,
            'minimo': minimo
        }
        
        print(f"{model_name:<60} | {media:<10.4f} | {desvio:<13.4f} | {maximo:<12.4f} | {minimo:<12.4f}")
    
    print("="*100)
    return stats


def plot_results(results, stats):
    """Cria gráficos de análise dos resultados"""
    
    model_names = list(results.keys())
    nomes_curtos = [
        'MQO', 'G.Tradicional', 'G.Cov_Treino', 'G.Agregada', 
        'Naive_Bayes', 'Friedman_0.25', 'Friedman_0.5', 'Friedman_0.75'
    ]
    
    cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    fig = plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 3, 1)
    dados_boxplot = [results[name] for name in model_names]
    box_plot = plt.boxplot(dados_boxplot, labels=nomes_curtos, patch_artist=True)
    
    for patch, cor in zip(box_plot['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)
    
    plt.title('Distribuição das Acurácias por Modelo')
    plt.ylabel('Acurácia')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    medias = [stats[name]['media'] for name in model_names]
    bars = plt.bar(nomes_curtos, medias, color=cores, alpha=0.8)
    plt.title('Acurácia Média por Modelo')
    plt.ylabel('Acurácia Média')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, valor in zip(bars, medias):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{valor:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 3, 3)
    desvios = [stats[name]['desvio'] for name in model_names]
    bars = plt.bar(nomes_curtos, desvios, color=cores, alpha=0.8)
    plt.title('Desvio Padrão das Acurácias')
    plt.ylabel('Desvio Padrão')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, valor in zip(bars, desvios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{valor:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 3, 4)
    maximos = [stats[name]['maximo'] for name in model_names]
    minimos = [stats[name]['minimo'] for name in model_names]
    
    x = np.arange(len(nomes_curtos))
    width = 0.35
    
    plt.bar(x - width/2, maximos, width, label='Máximo', alpha=0.8, color='lightgreen')
    plt.bar(x + width/2, minimos, width, label='Mínimo', alpha=0.8, color='lightcoral')
    
    plt.title('Valores Máximo e Mínimo por Modelo')
    plt.ylabel('Acurácia')
    plt.xticks(x, nomes_curtos, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    lambdas = [0, 0.25, 0.5, 0.75]
    modelos_friedman_nomes = [
        'Classificador Gaussiano Tradicional',
        'Classificador Gaussiano Regularizado (Friedman lambda=0.25)',
        'Classificador Gaussiano Regularizado (Friedman lambda=0.5)',
        'Classificador Gaussiano Regularizado (Friedman lambda=0.75)'
    ]
    acc_friedman = [stats[modelo]['media'] for modelo in modelos_friedman_nomes]
    
    plt.plot(lambdas, acc_friedman, 'bo-', linewidth=2, markersize=8)
    plt.title('Efeito da Regularização Friedman')
    plt.xlabel('Parâmetro λ')
    plt.ylabel('Acurácia Média')
    plt.grid(True, alpha=0.3)
    
    for x, y in zip(lambdas, acc_friedman):
        plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 3, 6)
    ranking = sorted(stats.items(), key=lambda x: x[1]['media'], reverse=True)
    nomes_ranking = [nome.split('(')[0].strip() for nome, _ in ranking]
    nomes_ranking = [nome[:20] + '...' if len(nome) > 20 else nome for nome in nomes_ranking]
    valores_ranking = [stat['media'] for _, stat in ranking]
    
    colors_ranking = plt.cm.Set3(np.linspace(0, 1, len(nomes_ranking)))
    bars = plt.barh(nomes_ranking, valores_ranking, color=colors_ranking, alpha=0.8)
    plt.title('Ranking dos Modelos (Acurácia Média)')
    plt.xlabel('Acurácia Média')
    plt.grid(True, alpha=0.3)
    
    for bar, valor in zip(bars, valores_ranking):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{valor:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig('resultados_classificadores.png', dpi=300, bbox_inches='tight')
    print("\nGráficos de resultados salvos como 'resultados_classificadores.png'")
    plt.show()
    
    return fig


if __name__ == "__main__":
    
    print("TRABALHO COMPUTACIONAL - CLASSIFICADORES GAUSSIANOS")
    print("="*80)
    
    try:
        data = np.loadtxt("../EMGsDataset.csv", delimiter=',')
    except FileNotFoundError:
        print("Arquivo EMGDataset.csv não encontrado!")
        print("Certifique-se de que o arquivo está no diretório correto.")
        exit()
    
    print(f"Dataset carregado: {data.shape}")
    
    if data.shape[0] == 3 and data.shape[1] == 50000:
        X_data = data[:2, :].T 
        y_data = data[2, :]  
    else:
        X_data = data[:, :-1]
        y_data = data[:, -1]
    
    # Filtrar apenas labels válidos (1-5)
    valid_mask = (y_data >= 1) & (y_data <= 5)
    X_data = X_data[valid_mask]
    y_data = y_data[valid_mask]
    
    N, p = X_data.shape
    C = 5
    
    print(f"Dados válidos: {N} amostras, {p} características, {C} classes")
    
    unique, counts = np.unique(y_data, return_counts=True)
    print("\nDistribuição das classes:")
    for cls, count in zip(unique, counts):
        print(f"Classe {int(cls)}: {count} amostras")
    
    print("\n2. VISUALIZAÇÃO DOS DADOS")
    print("-"*50)
    
    plt.figure(figsize=(10, 6))
    cores = ['red', 'blue', 'green', 'orange', 'purple']
    nomes = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']
    
    for i, (cls, cor, nome) in enumerate(zip(unique, cores, nomes)):
        mask = y_data == cls
        plt.scatter(X_data[mask, 0], X_data[mask, 1], 
                   c=cor, label=f'{int(cls)}-{nome}', alpha=0.6, s=20)
    
    plt.xlabel('Sensor 1 - Corrugador do Supercílio')
    plt.ylabel('Sensor 2 - Zigomático Maior')
    plt.title('Sinais EMG por Expressão Facial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Análise de separabilidade:")
    print("- Classes apresentam sobreposições")
    print("- Dados não são linearmente separáveis")
    print("- Classificadores probabilísticos são apropriados")
    
    print("\n3. ORGANIZAÇÃO DOS DADOS")
    print("-"*50)

    X_gauss = X_data.T  # (2 × N)
    y_gauss = y_data.reshape(1, -1)  # (1 × N)
    
    X_mqo = X_data  # (N × 2)
    y_mqo_onehot = one_hot_encode(y_data, C)  # (N × 5)
    
    print(f"Gaussianos - X: {X_gauss.shape}, y: {y_gauss.shape}")
    print(f"MQO - X: {X_mqo.shape}, y: {y_mqo_onehot.shape}")
    
    print("\n4. VALIDAÇÃO MONTE CARLO")
    print("-"*50)
    
    results = monte_carlo_validation(X_gauss, y_gauss, X_mqo, y_mqo_onehot, y_data, num_runs=500)
    
    stats = print_results_table(results)
    
    print("\n6. ANÁLISE DOS RESULTADOS")
    print("-"*50)
    
    best_model = max(stats.keys(), key=lambda k: stats[k]['media'])
    print(f"Melhor modelo: {best_model}")
    print(f"Acurácia média: {stats[best_model]['media']:.4f}")
    
    print("\nRanking dos modelos:")
    ranking = sorted(stats.items(), key=lambda x: x[1]['media'], reverse=True)
    for i, (model, stat) in enumerate(ranking):
        print(f"{i+1}. {model}: {stat['media']:.4f}")
    
    print("\n7. VISUALIZAÇÃO DOS RESULTADOS")
    print("-"*50)
    
    print("Mostrando gráfico de espalhamento dos dados EMG...")
    plt.figure(figsize=(10, 6))
    cores = ['red', 'blue', 'green', 'orange', 'purple']
    nomes = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']
    
    for i, (cls, cor, nome) in enumerate(zip(unique, cores, nomes)):
        mask = y_data == cls
        plt.scatter(X_data[mask, 0], X_data[mask, 1], 
                   c=cor, label=f'{int(cls)}-{nome}', alpha=0.6, s=20)
    
    plt.xlabel('Sensor 1 - Corrugador do Supercílio')
    plt.ylabel('Sensor 2 - Zigomático Maior')
    plt.title('Sinais EMG por Expressão Facial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Mostrando gráficos de análise dos resultados...")
    plot_results(results, stats)
    
    print("\nTrabalo concluído!")
    print("Gráficos exibidos: ")
    print("1. Espalhamento dos dados EMG")
    print("2. Análise comparativa dos classificadores")