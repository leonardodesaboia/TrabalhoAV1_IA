import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, List, Dict, Optional

class GaussianClassifierProblematico:
    """
    VERSÃO PEDAGÓGICA - DEMONSTRA OS PROBLEMAS DO MÉTODO TRADICIONAL
    
    Esta classe implementa o Classificador Gaussiano Tradicional SEM tratamentos
    de exceções para demonstrar onde e por que o método falha na prática.
    """
    
    def __init__(self, verbose=False):
        self.name = "Gaussiano Tradicional [PROBLEMÁTICO]"
        self.verbose = verbose
        self.classes = None
        self.C = None
        self.p = None
        self.N = None
        self.X = None
        self.n = None
        self.mu = None
        self.Sigma = None
        self.Sigma_det = None
        self.Sigma_inv = None
        self.P = None
        self.erro_detalhado = None
        self.falha_na_classe = None

    def fit(self, X_train, y_train):
        """
        Implementação SEM tratamento de exceções - VAI FALHAR propositalmente
        """
        if self.verbose:
            print(f"\n>>> TENTANDO TREINAR {self.name}")
        
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        if self.verbose:
            print(f"    Dimensões: {self.p} features, {self.N} amostras, {self.C} classes")
        
        # Separar dados por classe
        self.X = [X_train[:, y_train[0, :] == i] for i in self.classes]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        if self.verbose:
            print(f"    Amostras por classe: {self.n}")
        
        # Inicializar parâmetros
        self.mu = [None] * self.C
        self.Sigma = [None] * self.C
        self.Sigma_det = [None] * self.C
        self.Sigma_inv = [None] * self.C
        self.P = [None] * self.C
        
        # Calcular parâmetros para cada classe
        for i in range(self.C):
            if self.verbose:
                print(f"    Processando classe {int(self.classes[i])}...")
            
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
            
            # Calcular matriz de covariância tradicional
            self.Sigma[i] = np.cov(self.X[i])
            
            if self.verbose:
                print(f"      Matriz de covariância {self.Sigma[i].shape}")
                print(f"      Determinante: {np.linalg.det(self.Sigma[i]):.2e}")
                print(f"      Rank da matriz: {np.linalg.matrix_rank(self.Sigma[i])}")
                print(f"      Condition number: {np.linalg.cond(self.Sigma[i]):.2e}")
            
            # AQUI ESTÁ O PROBLEMA: Tentativa de inversão SEM verificação
            try:
                self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                if self.verbose:
                    print(f"      Tentando inverter matriz da classe {int(self.classes[i])}...")
                self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])  # VAI FALHAR!
                if self.verbose:
                    print(f"      ✓ Inversão bem-sucedida para classe {int(self.classes[i])}")
                
            except np.linalg.LinAlgError as e:
                self.erro_detalhado = str(e)
                self.falha_na_classe = int(self.classes[i])
                if self.verbose:
                    print(f"      ✗ FALHA na inversão da matriz da classe {int(self.classes[i])}")
                    print(f"      ✗ Erro: {e}")
                    print(f"      ✗ Motivo: Matriz singular (não invertível)")
                    print(f"      ✗ Isso acontece quando há poucas amostras ou features correlacionadas")
                raise e  # Re-lança a exceção para interromper o treinamento

    def predict(self, x_test):
        """Método de predição (nunca será usado na versão problemática)"""
        if self.Sigma_inv is None or any(S is None for S in self.Sigma_inv):
            raise RuntimeError("Modelo não foi treinado com sucesso devido a matriz singular")
        
        posteriori = [None] * self.C
        for i in range(self.C):
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]
            posteriori[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis
        return self.classes[np.argmax(posteriori)]


class GaussianClassifierCorrigido:
    """
    VERSÃO CORRIGIDA - DEMONSTRA AS SOLUÇÕES PARA OS PROBLEMAS
    
    Esta classe implementa as correções necessárias para tornar o 
    Classificador Gaussiano Tradicional robusto e utilizável na prática.
    """
    
    def __init__(self, covariance_type='traditional', lambda_reg=0, verbose=False):
        self.name = "Gaussiano Tradicional [CORRIGIDO]"
        self.covariance_type = covariance_type
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.classes = None
        self.C = None
        self.p = None
        self.N = None
        self.X = None
        self.n = None
        self.mu = None
        self.Sigma = None
        self.Sigma_det = None
        self.Sigma_inv = None
        self.P = None
        self.correcoes_aplicadas = []

    def fit(self, X_train, y_train):
        """
        Implementação COM tratamento robusto de exceções
        """
        if self.verbose:
            print(f"\n>>> TREINANDO {self.name}")
        
        self.classes = np.unique(y_train)
        self.C = len(self.classes)
        self.p, self.N = X_train.shape
        
        if self.verbose:
            print(f"    Dimensões: {self.p} features, {self.N} amostras, {self.C} classes")
        
        # Separar dados por classe
        self.X = [X_train[:, y_train[0, :] == i] for i in self.classes]
        self.n = [Xi.shape[1] for Xi in self.X]
        
        if self.verbose:
            print(f"    Amostras por classe: {self.n}")
        
        # Inicializar parâmetros
        self.mu = [None] * self.C
        self.Sigma = [None] * self.C
        self.Sigma_det = [None] * self.C
        self.Sigma_inv = [None] * self.C
        self.P = [None] * self.C
        
        # Calcular parâmetros para cada classe
        for i in range(self.C):
            if self.verbose:
                print(f"    Processando classe {int(self.classes[i])}...")
            
            self.mu[i] = np.mean(self.X[i], axis=1).reshape(self.p, 1)
            self.P[i] = self.n[i] / self.N
            
            # Calcular matriz de covariância baseada no tipo
            if self.covariance_type == 'traditional':
                self.Sigma[i] = self._compute_traditional_covariance(i)
            elif self.covariance_type == 'global':
                self.Sigma[i] = self._compute_global_covariance(X_train)
            elif self.covariance_type == 'pooled':
                self.Sigma[i] = self._compute_pooled_covariance()
            elif self.covariance_type == 'naive':
                self.Sigma[i] = self._compute_naive_covariance(i)
            elif self.covariance_type == 'friedman':
                self.Sigma[i] = self._compute_friedman_regularized(i)
            
            if self.verbose:
                print(f"      Determinante original: {np.linalg.det(self.Sigma[i]):.2e}")
                print(f"      Condition number original: {np.linalg.cond(self.Sigma[i]):.2e}")
            
            # SOLUÇÕES ROBUSTAS para problemas de inversão
            original_sigma = self.Sigma[i].copy()
            
            try:
                # Primeira tentativa: inversão direta
                self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])
                if self.verbose:
                    print(f"      ✓ Inversão direta bem-sucedida para classe {int(self.classes[i])}")
                
            except np.linalg.LinAlgError as e:
                if self.verbose:
                    print(f"      ⚠ Falha na inversão direta: {e}")
                    print(f"      → Aplicando CORREÇÃO 1: Micro-regularização")
                
                # CORREÇÃO 1: Adicionar pequena regularização na diagonal
                epsilon = 1e-6
                self.Sigma[i] = original_sigma + epsilon * np.eye(self.p)
                self.correcoes_aplicadas.append(f"Micro-regularização (ε={epsilon}) na classe {int(self.classes[i])}")
                
                try:
                    self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                    self.Sigma_inv[i] = np.linalg.inv(self.Sigma[i])
                    if self.verbose:
                        print(f"      ✓ Micro-regularização resolveu o problema!")
                    
                except np.linalg.LinAlgError:
                    if self.verbose:
                        print(f"      ⚠ Micro-regularização não foi suficiente")
                        print(f"      → Aplicando CORREÇÃO 2: Pseudo-inversa")
                    
                    # CORREÇÃO 2: Usar pseudo-inversa de Moore-Penrose
                    self.Sigma_det[i] = np.linalg.det(self.Sigma[i])
                    if self.Sigma_det[i] <= 0:
                        self.Sigma_det[i] = 1e-10  # Valor mínimo para log
                    
                    self.Sigma_inv[i] = np.linalg.pinv(self.Sigma[i])
                    self.correcoes_aplicadas.append(f"Pseudo-inversa na classe {int(self.classes[i])}")
                    if self.verbose:
                        print(f"      ✓ Pseudo-inversa aplicada com sucesso!")
            
            if self.verbose:
                print(f"      Determinante final: {self.Sigma_det[i]:.2e}")
                print(f"      Condition number final: {np.linalg.cond(self.Sigma[i]):.2e}")
        
        if self.verbose and self.correcoes_aplicadas:
            print(f"\n    CORREÇÕES APLICADAS:")
            for correcao in self.correcoes_aplicadas:
                print(f"      • {correcao}")
        elif self.verbose:
            print(f"\n    ✓ Nenhuma correção foi necessária - matrizes bem condicionadas")

    def _compute_traditional_covariance(self, class_idx):
        """Covariância tradicional para cada classe"""
        return np.cov(self.X[class_idx])

    def _compute_global_covariance(self, X_train):
        """Covariância global de todo conjunto de treino"""
        return np.cov(X_train)

    def _compute_pooled_covariance(self):
        """Covariância agregada (pooled)"""
        pooled_cov = np.zeros((self.p, self.p))
        for i in range(self.C):
            class_cov = np.cov(self.X[i])
            pooled_cov += (self.n[i] - 1) * class_cov
        return pooled_cov / (self.N - self.C)

    def _compute_naive_covariance(self, class_idx):
        """Covariância diagonal (Naive Bayes)"""
        cov_matrix = np.cov(self.X[class_idx])
        return np.diag(np.diag(cov_matrix))

    def _compute_friedman_regularized(self, class_idx):
        """Regularização de Friedman: (1-λ)Σ + λ*tr(Σ)/p*I"""
        sigma = np.cov(self.X[class_idx])
        trace_sigma = np.trace(sigma)
        identity = np.eye(self.p)
        return (1 - self.lambda_reg) * sigma + self.lambda_reg * (trace_sigma / self.p) * identity

    def predict(self, x_test):
        """Predição para uma amostra de teste"""
        posteriori = [None] * self.C
        for i in range(self.C):
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]
            posteriori[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis
        return self.classes[np.argmax(posteriori)]


class GaussianClassifier(GaussianClassifierCorrigido):
    """
    Classe principal que herda da versão corrigida
    Mantém compatibilidade com o código existente
    """
    def __init__(self, covariance_type='traditional', lambda_reg=0, verbose=False):
        super().__init__(covariance_type, lambda_reg, verbose)
        self.name = f"Gaussiano {covariance_type.title()}"

    def predict(self, x_test):
        """Predição para uma amostra de teste"""
        posteriori = [None] * self.C
        for i in range(self.C):
            d_mahalanobis = ((x_test - self.mu[i]).T @ self.Sigma_inv[i] @ (x_test - self.mu[i]))[0, 0]
            posteriori[i] = np.log(self.P[i]) - 0.5 * np.log(self.Sigma_det[i]) - 0.5 * d_mahalanobis
        return self.classes[np.argmax(posteriori)]


class MQOClassifier:
    def __init__(self, lambda_reg=1e-6):
        self.W = None
        self.classes = None
        self.lambda_reg = lambda_reg  # λ baixo para estabilidade
        
    def fit(self, X_train, y_train):
        """
        X_train: (N x p) - para MQO
        y_train: (N x C) - one-hot encoded
        """
        # Adicionar coluna de 1s para bias
        X_with_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        
        # MQO com λ baixo: W = (X^T X + λI)^(-1) X^T Y
        XtX = X_with_bias.T @ X_with_bias
        XtX_regularized = XtX + self.lambda_reg * np.eye(XtX.shape[0])
        self.W = np.linalg.inv(XtX_regularized) @ X_with_bias.T @ y_train
        self.classes = np.arange(1, y_train.shape[1] + 1)
        
    def predict(self, x_test):
        """Predição para uma amostra"""
        x_with_bias = np.hstack([1, x_test.flatten()])
        scores = x_with_bias @ self.W
        return self.classes[np.argmax(scores)]


def one_hot_encode(y, num_classes):
    """Converte labels para one-hot encoding - versão segura"""
    # Filtrar apenas labels válidos (1 a num_classes)
    valid_mask = (y >= 1) & (y <= num_classes)
    y_filtered = y[valid_mask]
    
    if len(y_filtered) != len(y):
        print(f"Atenção: {len(y) - len(y_filtered)} amostras com labels inválidos foram removidas")
        print(f"Labels únicos encontrados: {np.unique(y)}")
    
    y_one_hot = np.zeros((len(y_filtered), num_classes))
    for i, label in enumerate(y_filtered):
        y_one_hot[i, int(label) - 1] = 1
    return y_one_hot, valid_mask


def accuracy_score(y_true, y_pred):
    """Calcula acurácia"""
    return np.mean(y_true == y_pred)


# Carregamento dos dados
data = np.loadtxt("trabalho/EMGsDataset.csv", delimiter=',')

print("=" * 80)
print("ANÁLISE DE CLASSIFICAÇÃO EMG - SINAIS FACIAIS")
print("=" * 80)

# 1. ANÁLISE EXPLORATÓRIA DOS DADOS
print("\n1. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("-" * 50)

# Verificar dimensões reais dos dados
print(f"Dimensões originais do dataset: {data.shape}")

# Dados organizados como: (3 linhas x 50000 colunas)
# Linha 0: Sensor 1, Linha 1: Sensor 2, Linha 2: Labels
if data.shape[0] == 3:
    # Dados estão organizados como linhas
    X_data = data[:2, :].T  # Transpor para (N×p) -> (50000×2)
    y_data = data[2, :]     # Labels (50000,)
    N = data.shape[1]       # 50000 amostras
    p = 2                   # 2 features
else:
    # Dados estão organizados como colunas (formato original)
    X_data = data[:, :-1]   # Features (N×p)
    y_data = data[:, -1]    # Labels (N,)
    N, total_cols = data.shape
    p = total_cols - 1

C = 5  # número de classes

print(f"Dimensões do dataset: {N} observações, {p} features")
print(f"Número de classes: {C}")
print(f"Classes: 1-Neutro, 2-Sorriso, 3-Sobrancelhas, 4-Surpreso, 5-Rabugento")
print(f"Shape de X_data: {X_data.shape}")
print(f"Shape de y_data: {y_data.shape}")

# Verificar distribuição das classes e filtrar dados inválidos
print(f"\nVerificando labels únicos: {np.unique(y_data)}")
print(f"Range esperado: 1 a {C}")

# Filtrar apenas amostras com labels válidos
valid_labels_mask = (y_data >= 1) & (y_data <= C)
X_data = X_data[valid_labels_mask]
y_data = y_data[valid_labels_mask]
N = len(y_data)

print(f"Amostras após filtragem: {N}")

unique_classes, counts = np.unique(y_data, return_counts=True)
print(f"\nDistribuição das classes:")
for cls, count in zip(unique_classes, counts):
    print(f"Classe {int(cls)}: {count} amostras ({count/N*100:.1f}%)")

print(f"\nEstatísticas das features:")
for i in range(p):
    print(f"Feature {i+1}: média={np.mean(X_data[:, i]):.2f}, std={np.std(X_data[:, i]):.2f}")

# ORGANIZAÇÃO DOS DADOS (movido para cá para uso na demonstração)
# Para modelos gaussianos: X ∈ R^(p×N), Y ∈ R^(C×N)
X_gauss = X_data.T  # (p×N)
y_gauss = y_data.reshape(1, -1)  # (1×N)

# Para MQO: X ∈ R^(N×p), Y ∈ R^(N×C)
X_mqo = X_data  # (N×p)
y_mqo_onehot, _ = one_hot_encode(y_data, C)  # (N×C) - dados já foram filtrados

# 2. VISUALIZAÇÃO INICIAL DOS DADOS
print(f"\n2. VISUALIZAÇÃO E ANÁLISE DE SEPARABILIDADE")
print("-" * 50)

plt.figure(1, figsize=(12, 8))

# Gráfico de espalhamento
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

print("""
ANÁLISE DE SEPARABILIDADE:

1. DISTRIBUIÇÃO ESPACIAL: As classes mostram sobreposições consideráveis
2. SEPARABILIDADE LINEAR: Dados não são linearmente separáveis
3. COMPLEXIDADE: Classificadores não-lineares podem ser mais eficazes
4. RUÍDO: Sinais EMG apresentam variabilidade natural
5. CORRELAÇÃO: Features podem ter dependências que Naive Bayes ignora

CARACTERÍSTICAS DO MODELO IDEAL:
- Capturar distribuições gaussianas de cada classe
- Lidar com sobreposições através de probabilidades
- Considerar correlações entre sensores
- Ser robusto a outliers e ruído dos sinais EMG
""")

# 3. DEMONSTRAÇÃO PEDAGÓGICA: PROBLEMA vs SOLUÇÃO
print(f"\n3. DEMONSTRAÇÃO PEDAGÓGICA: GAUSSIANO PROBLEMÁTICO vs CORRIGIDO")
print("=" * 80)

print("""
OBJETIVO PEDAGÓGICO:
Demonstrar por que o Classificador Gaussiano Tradicional falha na prática
e como corrigir esses problemas com técnicas robustas.

PROBLEMAS COMUNS:
1. Matrizes de covariância singulares (não invertíveis)
2. Poucas amostras por classe
3. Features altamente correlacionadas
4. Problemas de condicionamento numérico

SOLUÇÕES:
1. Micro-regularização: Σ + εI
2. Pseudo-inversa de Moore-Penrose
3. Regularização de Friedman
4. Covariância agregada (pooled)
""")

# Preparar dados para demonstração
print(f"\nPreparando dados para demonstração...")
print(f"Usando uma amostra pequena para forçar problemas de singularidade")

# Usar uma amostra muito pequena para cada classe (força problemas)
np.random.seed(42)  # Para reprodutibilidade
idx_demo = np.random.permutation(N)[:100]  # Apenas 100 amostras

X_demo_gauss = X_gauss[:, idx_demo]
y_demo_gauss = y_gauss[:, idx_demo]

print(f"Dados para demonstração: {X_demo_gauss.shape[1]} amostras")

# Verificar distribuição das classes na amostra pequena
unique_demo, counts_demo = np.unique(y_demo_gauss[0, :], return_counts=True)
print(f"Distribuição na amostra pequena:")
for cls, count in zip(unique_demo, counts_demo):
    print(f"  Classe {int(cls)}: {count} amostras")

print(f"\nCom tão poucas amostras, algumas classes terão matrizes de covariância singulares!")

# PARTE A: TENTATIVA COM VERSÃO PROBLEMÁTICA
print(f"\n" + "="*80)
print(f"PARTE A: TESTANDO VERSÃO PROBLEMÁTICA (SEM CORREÇÕES)")
print(f"="*80)

try:
    gauss_problematico = GaussianClassifierProblematico(verbose=True)  # Mostrar detalhes apenas aqui
    gauss_problematico.fit(X_demo_gauss, y_demo_gauss)
    print(f"\n⚠ INESPERADO: O modelo problemático funcionou!")
    print(f"   Isso pode acontecer por sorte com a amostra específica.")
    sucesso_problematico = True
    
except Exception as e:
    print(f"\n✓ COMPORTAMENTO ESPERADO: Falha conforme previsto!")
    print(f"   Classe que falhou: {gauss_problematico.falha_na_classe if hasattr(gauss_problematico, 'falha_na_classe') else 'N/A'}")
    print(f"   Erro específico: {str(e)}")
    print(f"   Tipo do erro: {type(e).__name__}")
    sucesso_problematico = False

# PARTE B: APLICAÇÃO DA VERSÃO CORRIGIDA
print(f"\n" + "="*80)
print(f"PARTE B: TESTANDO VERSÃO CORRIGIDA (COM SOLUÇÕES ROBUSTAS)")
print(f"="*80)

try:
    gauss_corrigido = GaussianClassifierCorrigido(covariance_type='traditional', verbose=True)  # Mostrar detalhes apenas aqui
    gauss_corrigido.fit(X_demo_gauss, y_demo_gauss)
    print(f"\n✓ SUCESSO: Versão corrigida funcionou perfeitamente!")
    sucesso_corrigido = True
    
    # Teste de predição
    x_teste = X_demo_gauss[:, 0].reshape(-1, 1)
    pred = gauss_corrigido.predict(x_teste)
    print(f"   Teste de predição: Classe predita = {pred}")
    
except Exception as e:
    print(f"\n✗ FALHA INESPERADA na versão corrigida: {str(e)}")
    sucesso_corrigido = False

# PARTE C: COMPARAÇÃO E ANÁLISE
print(f"\n" + "="*80)
print(f"PARTE C: ANÁLISE COMPARATIVA")
print(f"="*80)

print(f"RESULTADOS DA DEMONSTRAÇÃO:")
print(f"  • Versão Problemática: {'✓ Funcionou' if sucesso_problematico else '✗ Falhou (esperado)'}")
print(f"  • Versão Corrigida:    {'✓ Funcionou' if sucesso_corrigido else '✗ Falhou'}")

print(f"\nCONCLUSÕES PEDAGÓGICAS:")

if not sucesso_problematico and sucesso_corrigido:
    print(f"✓ DEMONSTRAÇÃO PERFEITA:")
    print(f"  1. Versão tradicional falhou devido a matrizes singulares")
    print(f"  2. Versão corrigida resolveu o problema com técnicas robustas")
    print(f"  3. Isso mostra a importância de implementações numericamente estáveis")

elif sucesso_problematico and sucesso_corrigido:
    print(f"⚠ DEMONSTRAÇÃO PARCIAL:")
    print(f"  1. Por sorte, a amostra não causou problemas na versão tradicional")
    print(f"  2. Isso pode acontecer com dados bem condicionados")
    print(f"  3. Em datasets reais, problemas são mais frequentes")

else:
    print(f"⚠ RESULTADO INESPERADO:")
    print(f"  1. Ambas versões falharam ou ambas funcionaram")
    print(f"  2. Pode indicar problemas nos dados ou implementação")

print(f"\nIMPLICAÇÕES PRÁTICAS:")
print(f"• SEMPRE usar implementações robustas em produção")
print(f"• Tratar matrizes singulares é fundamental")
print(f"• Micro-regularização é uma técnica simples e eficaz")
print(f"• Pseudo-inversa é útil quando regularização não basta")
print(f"• Verificar condition number das matrizes")

# 4. ORGANIZAÇÃO DOS DADOS PARA VALIDAÇÃO COMPLETA
print(f"\n4. ORGANIZAÇÃO DOS DADOS PARA VALIDAÇÃO COMPLETA")
print("-" * 50)

print(f"Organização para Gaussianos:")
print(f"X_gauss: {X_gauss.shape}")
print(f"y_gauss: {y_gauss.shape}")
print(f"\nOrganização para MQO:")
print(f"X_mqo: {X_mqo.shape}")
print(f"y_mqo_onehot: {y_mqo_onehot.shape}")

# 5. VALIDAÇÃO MONTE CARLO
print(f"\n5. VALIDAÇÃO MONTE CARLO (R = 50 rodadas)")
print("-" * 50)

rodadas = 50

# Armazenamento dos resultados
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

# Contadores para problemas detectados
problemas_detectados = {
    'mqo_falhas': 0,
    'gauss_correcoes': 0,
    'gauss_falhas_totais': 0
}

for r in range(rodadas):
    # Mostrar progresso a cada 5 rodadas + flush para aparecer imediatamente
    if r % 5 == 0:
        print(f"{r}", end=" ", flush=True)
    
    # Embaralhar dados
    idx = np.random.permutation(N)
    
    # Particionamento 80/20
    split_idx = int(N * 0.8)
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]
    
    # Dados de treino e teste para gaussianos
    X_train_g = X_gauss[:, train_idx]
    y_train_g = y_gauss[:, train_idx]
    X_test_g = X_gauss[:, test_idx]
    y_test_g = y_gauss[:, test_idx]
    
    # Dados de treino e teste para MQO
    X_train_m = X_mqo[train_idx, :]
    y_train_m = y_mqo_onehot[train_idx, :]
    X_test_m = X_mqo[test_idx, :]
    y_test_m = y_data[test_idx]
    
    # 1. MQO Tradicional
    try:
        mqo_clf = MQOClassifier()
        mqo_clf.fit(X_train_m, y_train_m)
        
        mqo_predictions = []
        for i in range(len(X_test_m)):
            pred = mqo_clf.predict(X_test_m[i, :])
            mqo_predictions.append(pred)
        
        mqo_acc = accuracy_score(y_test_m, np.array(mqo_predictions))
        acuracia_resultados['mqo_tradicional'].append(mqo_acc)
    except Exception as e:
        problemas_detectados['mqo_falhas'] += 1
        acuracia_resultados['mqo_tradicional'].append(0.0)
    
    # 2. Classificador Gaussiano Tradicional (COM LOGGING DE PROBLEMAS)
    try:
        # Usando a versão corrigida mas rastreando problemas (SEM verbose para não poluir)
        gauss_trad = GaussianClassifierCorrigido(covariance_type='traditional', verbose=False)
        gauss_trad.fit(X_train_g, y_train_g)
        
        # Verificar se correções foram aplicadas
        if gauss_trad.correcoes_aplicadas:
            problemas_detectados['gauss_correcoes'] += 1
        
        gauss_trad_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_trad.predict(x_test)
            gauss_trad_predictions.append(pred)
        
        gauss_trad_acc = accuracy_score(y_test_g[0, :], np.array(gauss_trad_predictions))
        acuracia_resultados['gauss_tradicional'].append(gauss_trad_acc)
        
    except Exception as e:
        problemas_detectados['gauss_falhas_totais'] += 1
        acuracia_resultados['gauss_tradicional'].append(0.0)
    
    # 3. Classificador Gaussiano (Cov. de todo cj. treino)
    try:
        gauss_global = GaussianClassifier(covariance_type='global')
        gauss_global.fit(X_train_g, y_train_g)
        
        gauss_global_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_global.predict(x_test)
            gauss_global_predictions.append(pred)
        
        gauss_global_acc = accuracy_score(y_test_g[0, :], np.array(gauss_global_predictions))
        acuracia_resultados['gauss_global'].append(gauss_global_acc)
    except:
        acuracia_resultados['gauss_global'].append(0.0)
    
    # 4. Classificador Gaussiano (Cov. Agregada)
    try:
        gauss_pooled = GaussianClassifier(covariance_type='pooled')
        gauss_pooled.fit(X_train_g, y_train_g)
        
        gauss_pooled_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_pooled.predict(x_test)
            gauss_pooled_predictions.append(pred)
        
        gauss_pooled_acc = accuracy_score(y_test_g[0, :], np.array(gauss_pooled_predictions))
        acuracia_resultados['gauss_pooled'].append(gauss_pooled_acc)
    except:
        acuracia_resultados['gauss_pooled'].append(0.0)
    
    # 5. Classificador de Bayes Ingênuo
    try:
        naive_bayes = GaussianClassifier(covariance_type='naive')
        naive_bayes.fit(X_train_g, y_train_g)
        
        naive_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = naive_bayes.predict(x_test)
            naive_predictions.append(pred)
        
        naive_acc = accuracy_score(y_test_g[0, :], np.array(naive_predictions))
        acuracia_resultados['naive_bayes'].append(naive_acc)
    except:
        acuracia_resultados['naive_bayes'].append(0.0)
    
    # 6. Classificador Gaussiano Regularizado (Friedman λ=0.25)
    try:
        gauss_f025 = GaussianClassifier(covariance_type='friedman', lambda_reg=0.25)
        gauss_f025.fit(X_train_g, y_train_g)
        
        f025_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_f025.predict(x_test)
            f025_predictions.append(pred)
        
        f025_acc = accuracy_score(y_test_g[0, :], np.array(f025_predictions))
        acuracia_resultados['gauss_friedman_025'].append(f025_acc)
    except:
        acuracia_resultados['gauss_friedman_025'].append(0.0)
    
    # 7. Classificador Gaussiano Regularizado (Friedman λ=0.5)
    try:
        gauss_f050 = GaussianClassifier(covariance_type='friedman', lambda_reg=0.5)
        gauss_f050.fit(X_train_g, y_train_g)
        
        f050_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_f050.predict(x_test)
            f050_predictions.append(pred)
        
        f050_acc = accuracy_score(y_test_g[0, :], np.array(f050_predictions))
        acuracia_resultados['gauss_friedman_050'].append(f050_acc)
    except:
        acuracia_resultados['gauss_friedman_050'].append(0.0)
    
    # 8. Classificador Gaussiano Regularizado (Friedman λ=0.75)
    try:
        gauss_f075 = GaussianClassifier(covariance_type='friedman', lambda_reg=0.75)
        gauss_f075.fit(X_train_g, y_train_g)
        
        f075_predictions = []
        for i in range(X_test_g.shape[1]):
            x_test = X_test_g[:, i].reshape(-1, 1)
            pred = gauss_f075.predict(x_test)
            f075_predictions.append(pred)
        
        f075_acc = accuracy_score(y_test_g[0, :], np.array(f075_predictions))
        acuracia_resultados['gauss_friedman_075'].append(f075_acc)
    except:
        acuracia_resultados['gauss_friedman_075'].append(0.0)

print(f"\n\nSimulações concluídas!")

# Relatório de problemas detectados durante a validação
print(f"\nRELATÓRIO DE ROBUSTEZ DURANTE A VALIDAÇÃO:")
print(f"  • MQO - Falhas totais: {problemas_detectados['mqo_falhas']}/{rodadas} ({problemas_detectados['mqo_falhas']/rodadas*100:.1f}%)")
print(f"  • Gaussiano - Correções aplicadas: {problemas_detectados['gauss_correcoes']}/{rodadas} ({problemas_detectados['gauss_correcoes']/rodadas*100:.1f}%)")
print(f"  • Gaussiano - Falhas totais: {problemas_detectados['gauss_falhas_totais']}/{rodadas} ({problemas_detectados['gauss_falhas_totais']/rodadas*100:.1f}%)")

if problemas_detectados['gauss_correcoes'] > 0:
    print(f"\n✓ As correções no Classificador Gaussiano foram aplicadas em {problemas_detectados['gauss_correcoes']} rodadas")
    print(f"  Isso demonstra a importância das técnicas de estabilização numérica!")
else:
    print(f"\n→ Nenhuma correção foi necessária nas {rodadas} rodadas")
    print(f"  Isso indica que os dados EMG estão bem condicionados")

# 6. CÁLCULO DAS ESTATÍSTICAS E APRESENTAÇÃO DOS RESULTADOS
print(f"\n6. RESULTADOS FINAIS - ESTATÍSTICAS DAS ACURÁCIAS")
print("=" * 80)

# Nomes dos modelos para exibição
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

chaves_modelos = [
    'mqo_tradicional',
    'gauss_tradicional',
    'gauss_global',
    'gauss_pooled',
    'naive_bayes',
    'gauss_friedman_025',
    'gauss_friedman_050',
    'gauss_friedman_075'
]

# Cabeçalho da tabela
print(f"{'Modelos':<50} | {'Média':<10} | {'Desvio-Padrão':<13} | {'Maior Valor':<12} | {'Menor Valor':<12}")
print("-" * 50 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 13)

# Calcular e exibir estatísticas para cada modelo
estatisticas = {}

for i, (nome, chave) in enumerate(zip(nomes_modelos, chaves_modelos)):
    acc_valores = np.array(acuracia_resultados[chave])
    
    media = np.mean(acc_valores)
    desvio = np.std(acc_valores)
    maximo = np.max(acc_valores)
    minimo = np.min(acc_valores)
    
    estatisticas[chave] = {
        'media': media,
        'desvio': desvio,
        'maximo': maximo,
        'minimo': minimo
    }
    
    print(f"{nome:<50} | {media:<10.4f} | {desvio:<13.4f} | {maximo:<12.4f} | {minimo:<12.4f}")

print("=" * 80)

# 7. ANÁLISE DOS RESULTADOS
print(f"\n7. ANÁLISE DOS RESULTADOS")
print("-" * 50)

# Encontrar o melhor modelo (maior acurácia média)
melhor_modelo = max(estatisticas.keys(), key=lambda k: estatisticas[k]['media'])
idx_melhor = chaves_modelos.index(melhor_modelo)
nome_melhor = nomes_modelos[idx_melhor]

print(f"MELHOR MODELO: {nome_melhor}")
print(f"Acurácia média: {estatisticas[melhor_modelo]['media']:.4f}")
print(f"Desvio padrão: {estatisticas[melhor_modelo]['desvio']:.4f}")

# Comparação entre modelos Gaussianos
print(f"\nRANKING DOS MODELOS (por acurácia média):")
ranking = sorted(estatisticas.items(), key=lambda x: x[1]['media'], reverse=True)
for i, (chave, stats) in enumerate(ranking):
    idx = chaves_modelos.index(chave)
    nome = nomes_modelos[idx]
    print(f"{i+1}. {nome}: {stats['media']:.4f}")

# 8. VISUALIZAÇÃO DOS RESULTADOS
print(f"\n8. GRÁFICOS COMPARATIVOS")
print("-" * 50)

plt.figure(2, figsize=(15, 10))

# Gráfico 1: Boxplot das acurácias
plt.subplot(2, 2, 1)
dados_boxplot = [acuracia_resultados[chave] for chave in chaves_modelos]
nomes_curtos = ['MQO', 'G.Tradicional', 'G.cj_treino', 'G.Agregada', 'Naive-Bayes', 'Friedman-0.25', 'Friedman-0.5', 'Friedman-0.75']
cores_box = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
box_plot = plt.boxplot(dados_boxplot, labels=nomes_curtos, patch_artist=True)
for patch, cor in zip(box_plot['boxes'], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
plt.title('Distribuição das Acurácias por Modelo')
plt.ylabel('Acurácia')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 2: Comparação de médias
plt.subplot(2, 2, 2)
medias_acc = [estatisticas[chave]['media'] for chave in chaves_modelos]
bars = plt.bar(nomes_curtos, medias_acc, color=cores_box, alpha=0.8)
plt.title('Acurácia Média por Modelo')
plt.ylabel('Acurácia Média')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bar, valor in zip(bars, medias_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{valor:.3f}', ha='center', va='bottom', fontsize=9)

# Gráfico 3: Desvio padrão
plt.subplot(2, 2, 3)
desvios_acc = [estatisticas[chave]['desvio'] for chave in chaves_modelos]
bars = plt.bar(nomes_curtos, desvios_acc, color=cores_box, alpha=0.8)
plt.title('Desvio Padrão das Acurácias')
plt.ylabel('Desvio Padrão')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Adicionar valores nas barras
for bar, valor in zip(bars, desvios_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{valor:.3f}', ha='center', va='bottom', fontsize=9)

# Gráfico 4: Efeito da regularização Friedman
plt.subplot(2, 2, 4)
lambdas = [0, 0.25, 0.5, 0.75]
modelos_friedman = ['gauss_tradicional', 'gauss_friedman_025', 'gauss_friedman_050', 'gauss_friedman_075']
acc_friedman = [estatisticas[modelo]['media'] for modelo in modelos_friedman]

plt.plot(lambdas, acc_friedman, 'bo-', linewidth=2, markersize=8)
plt.title('Efeito da Regularização Friedman')
plt.xlabel('Parâmetro λ')
plt.ylabel('Acurácia Média')
plt.grid(True, alpha=0.3)

# Adicionar valores nos pontos
for x, y in zip(lambdas, acc_friedman):
    plt.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# MOSTRAR TODOS OS GRÁFICOS SIMULTANEAMENTE
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

print("\n" + "="*80)
print("RESUMO DA DEMONSTRAÇÃO PEDAGÓGICA")
print("="*80)

print("""
PROBLEMA DEMONSTRADO:
✗ Classificador Gaussiano Tradicional pode falhar com matrizes singulares
✗ Isso acontece com poucas amostras ou features correlacionadas
✗ Implementações ingênuas são vulneráveis a esses problemas

SOLUÇÕES IMPLEMENTADAS:
✓ Micro-regularização: Σ + εI (adiciona estabilidade)
✓ Pseudo-inversa: Moore-Penrose (quando inversão falha)
✓ Regularização de Friedman: (1-λ)Σ + λ(tr(Σ)/p)I
✓ Covariância agregada: pooling entre classes

LIÇÕES APRENDIDAS:
1. SEMPRE implementar verificações de robustez numérica
2. Tratar exceções específicas (LinAlgError)
3. Usar pseudo-inversa como fallback
4. Monitorar condition number das matrizes
5. Regularização previne muitos problemas

RELEVÂNCIA PRÁTICA:
• Dados reais frequentemente têm problemas de condicionamento
• Sinais EMG são ruidosos e podem ter correlações altas
• Implementações robustas são essenciais em produção
• Técnicas de regularização melhoram generalização

Este código demonstra como transformar um algoritmo teoricamente correto
em uma implementação praticamente utilizável e robusta.
""")

print("="*80)