import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def predicao(X, beta):
    return X @ beta

def modelo_ridge(X, y, lambda_val):
    """Implementa regressão Ridge: β = (X^T X + λI)^(-1) X^T y"""
    I = np.eye(X.shape[1])
    I[0, 0] = 0 
    beta = pinv(X.T @ X + lambda_val * I) @ X.T @ y
    return beta

data = np.loadtxt("trabalho/aerogerador.dat", delimiter=None)

print("="*80)
print("ANÁLISE DE REGRESSÃO - DADOS DO AEROGERADOR")
print("="*80)

print("\n1. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("-" * 50)

N, total_cols = data.shape
p = total_cols - 1  # Variáveis regressoras

print(f"Dimensões do dataset: {N} observações, {p} variáveis regressoras")
print(f"Variáveis regressoras: {p} colunas")
print(f"Variável dependente: 1 coluna")


X_ = data[:, :-1]
y = data[:, -1:]

print(f"\nEstatísticas da variável dependente:")
print(f"Média: {np.mean(y):.4f}")
print(f"Desvio padrão: {np.std(y):.4f}")
print(f"Mínimo: {np.min(y):.4f}")
print(f"Máximo: {np.max(y):.4f}")


plt.figure(1, figsize=(10, 6))
if p == 2:  
    plt.scatter(X_[:, 0], y[:, 0], alpha=0.6, label='Var 1 vs Y', color='blue')
    plt.scatter(X_[:, 1], y[:, 0], alpha=0.6, label='Var 2 vs Y', color='red')
    plt.xlabel("Variáveis Regressoras")
    plt.ylabel("Variável Dependente")
    plt.title("Gráfico de Espalhamento dos Dados Originais")
    plt.legend()
elif p == 1:
    plt.scatter(X_[:, 0], y[:, 0], edgecolor='k', alpha=0.6, color='blue')
    plt.xlabel("Variável Regressora")
    plt.ylabel("Variável Dependente")
    plt.title("Gráfico de Espalhamento dos Dados Originais")
else:
    plt.scatter(X_[:, 0], y[:, 0], alpha=0.6, edgecolor='k', color='blue')
    plt.xlabel("Primeira Variável Regressora")
    plt.ylabel("Variável Dependente")
    plt.title("Gráfico de Espalhamento dos Dados Originais")

plt.grid(True, alpha=0.3)
plt.tight_layout()

print(f"\n2. DISCUSSÃO SOBRE CARACTERÍSTICAS DO MODELO IDEAL")
print("-" * 50)
print("""
CARACTERÍSTICAS DE UM BOM MODELO DE REGRESSÃO:

1. LINEARIDADE: O modelo deve capturar relações lineares entre X e y
2. GENERALIZAÇÃO: Deve funcionar bem em dados não vistos durante o treino
3. SIMPLICIDADE vs COMPLEXIDADE: Balance entre bias e variância
4. ROBUSTEZ: Resistente a outliers e ruído nos dados
5. INTERPRETABILIDADE: Coeficientes com significado prático

A regularização Ridge ajuda a:
- Reduzir overfitting
- Melhorar estabilidade numérica  
- Controlar a complexidade do modelo
""")


print(f"\n3. ORGANIZAÇÃO DOS DADOS")
print("-" * 50)

X = np.hstack((np.ones((N, 1)), X_))

print(f"Matriz X (com intercepto): {X.shape}")
print(f"Vetor y: {y.shape}")
print(f"Primeiras 5 linhas de X:")
print(X[:5, :])
print(f"Primeiros 5 valores de y: {y[:5, 0]}")

print(f"\n4. VALIDAÇÃO MONTE CARLO (R = 500 rodadas)")
print("-" * 50)

rodadas = 500

rss_resultados = {
    'media_var_dep': [],
    'mqo_tradicional': [],
    'ridge_025': [],
    'ridge_050': [],
    'ridge_075': [],
    'ridge_100': []
}

print("Gerando gráfico de comparação de todos os modelos...")

sample_size = min(200, N)
sample_idx = np.random.choice(N, sample_size, replace=False)
X_sample = X[sample_idx, :]
y_sample = y[sample_idx, :]

beta_media_vis = np.zeros((X.shape[1], 1))
beta_media_vis[0, 0] = np.mean(y_sample)

beta_mqo_vis = pinv(X_sample.T @ X_sample) @ X_sample.T @ y_sample
beta_ridge_025_vis = modelo_ridge(X_sample, y_sample, 0.25)
beta_ridge_050_vis = modelo_ridge(X_sample, y_sample, 0.5)
beta_ridge_075_vis = modelo_ridge(X_sample, y_sample, 0.75)
beta_ridge_100_vis = modelo_ridge(X_sample, y_sample, 1.0)

y_pred_media_vis = predicao(X_sample, beta_media_vis)
y_pred_mqo_vis = predicao(X_sample, beta_mqo_vis)
y_pred_025_vis = predicao(X_sample, beta_ridge_025_vis)
y_pred_050_vis = predicao(X_sample, beta_ridge_050_vis)
y_pred_075_vis = predicao(X_sample, beta_ridge_075_vis)
y_pred_100_vis = predicao(X_sample, beta_ridge_100_vis)

plt.figure(2, figsize=(16, 8))

if p == 1:

    sort_idx = np.argsort(X_sample[:, 1])
    X_sorted = X_sample[sort_idx, 1]
    y_sorted = y_sample[sort_idx, 0]
    
    plt.scatter(X_sorted, y_sorted, color='black', alpha=0.7, s=20, label='Dados Reais', zorder=10)
    
    plt.plot(X_sorted, y_pred_media_vis[sort_idx, 0], '--', linewidth=4, color='red', label='Média da variável dependente', alpha=0.8)
    plt.plot(X_sorted, y_pred_mqo_vis[sort_idx, 0], '-', linewidth=3, color='blue', label='MQO tradicional', alpha=0.9)
    plt.plot(X_sorted, y_pred_025_vis[sort_idx, 0], '-', linewidth=2.5, color='green', label='MQO regularizado (0,25)', alpha=0.9)
    plt.plot(X_sorted, y_pred_050_vis[sort_idx, 0], '-', linewidth=2.5, color='orange', label='MQO regularizado (0,5)', alpha=0.9)
    plt.plot(X_sorted, y_pred_075_vis[sort_idx, 0], '-', linewidth=2.5, color='purple', label='MQO regularizado (0,75)', alpha=0.9)
    plt.plot(X_sorted, y_pred_100_vis[sort_idx, 0], '-', linewidth=2.5, color='brown', label='MQO regularizado (1)', alpha=0.9)
    
    plt.xlabel('Variável Regressora', fontsize=12)
    plt.ylabel('Variável Dependente', fontsize=12)
    plt.title(' COMPARAÇÃO DE TODOS OS MODELOS DE REGRESSÃO ', fontsize=16, fontweight='bold')
    
else:
    plt.scatter(y_sample[:, 0], y_pred_media_vis[:, 0], alpha=0.7, s=40, color='red', label='Média da variável dependente')
    plt.scatter(y_sample[:, 0], y_pred_mqo_vis[:, 0], alpha=0.7, s=40, color='blue', label='MQO tradicional')
    plt.scatter(y_sample[:, 0], y_pred_025_vis[:, 0], alpha=0.7, s=40, color='green', label='MQO regularizado (0,25)')
    plt.scatter(y_sample[:, 0], y_pred_050_vis[:, 0], alpha=0.7, s=40, color='orange', label='MQO regularizado (0,5)')
    plt.scatter(y_sample[:, 0], y_pred_075_vis[:, 0], alpha=0.7, s=40, color='purple', label='MQO regularizado (0,75)')
    plt.scatter(y_sample[:, 0], y_pred_100_vis[:, 0], alpha=0.7, s=40, color='brown', label='MQO regularizado (1)')
    
    min_val = min(np.min(y_sample), 
                  np.min([y_pred_media_vis, y_pred_mqo_vis, y_pred_025_vis, 
                         y_pred_050_vis, y_pred_075_vis, y_pred_100_vis]))
    max_val = max(np.max(y_sample), 
                  np.max([y_pred_media_vis, y_pred_mqo_vis, y_pred_025_vis, 
                         y_pred_050_vis, y_pred_075_vis, y_pred_100_vis]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Predição Perfeita')
    
    plt.xlabel('Valores Reais', fontsize=12)
    plt.ylabel('Valores Preditos', fontsize=12)
    plt.title(' COMPARAÇÃO DE TODOS OS MODELOS - Predito vs Real ', fontsize=16, fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("Executando simulações...")
print("Progresso: ", end="")

for r in range(rodadas):
    if r % 50 == 0:
        print(f"{r}", end=" ")
    
    idx = np.random.permutation(N)
    Xr = X[idx, :]
    yr = y[idx, :]
    
    split_idx = int(N * 0.8)
    X_treino = Xr[:split_idx, :]
    y_treino = yr[:split_idx, :]
    X_teste = Xr[split_idx:, :]
    y_teste = yr[split_idx:, :]
    
    # TREINAMENTO DOS MODELOS
    
    beta_media = np.zeros((X.shape[1], 1))
    beta_media[0, 0] = np.mean(y_treino)  
    
    beta_mqo = pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    
    beta_ridge_025 = modelo_ridge(X_treino, y_treino, 0.25)
    beta_ridge_050 = modelo_ridge(X_treino, y_treino, 0.5)
    beta_ridge_075 = modelo_ridge(X_treino, y_treino, 0.75)
    beta_ridge_100 = modelo_ridge(X_treino, y_treino, 1.0)
    
    # AVALIAÇÃO DOS MODELOS (RSS)
    y_pred_media = predicao(X_teste, beta_media)
    rss_media = np.sum((y_teste - y_pred_media) ** 2)
    rss_resultados['media_var_dep'].append(rss_media)
    
    y_pred_mqo = predicao(X_teste, beta_mqo)
    rss_mqo = np.sum((y_teste - y_pred_mqo) ** 2)
    rss_resultados['mqo_tradicional'].append(rss_mqo)
    
    y_pred_025 = predicao(X_teste, beta_ridge_025)
    rss_025 = np.sum((y_teste - y_pred_025) ** 2)
    rss_resultados['ridge_025'].append(rss_025)
    
    y_pred_050 = predicao(X_teste, beta_ridge_050)
    rss_050 = np.sum((y_teste - y_pred_050) ** 2)
    rss_resultados['ridge_050'].append(rss_050)
    
    y_pred_075 = predicao(X_teste, beta_ridge_075)
    rss_075 = np.sum((y_teste - y_pred_075) ** 2)
    rss_resultados['ridge_075'].append(rss_075)
    
    y_pred_100 = predicao(X_teste, beta_ridge_100)
    rss_100 = np.sum((y_teste - y_pred_100) ** 2)
    rss_resultados['ridge_100'].append(rss_100)

print("\n\nSimulações concluídas!")

print(f"\n5. RESULTADOS FINAIS - ESTATÍSTICAS DOS RSS")
print("="*80)

nomes_modelos = [
    'Média da variável dependente',
    'MQO tradicional',
    'MQO regularizado (0,25)',
    'MQO regularizado (0,5)',
    'MQO regularizado (0,75)',
    'MQO regularizado (1)'
]

chaves_modelos = [
    'media_var_dep',
    'mqo_tradicional', 
    'ridge_025',
    'ridge_050',
    'ridge_075',
    'ridge_100'
]

print(f"{'Modelos':<30} | {'Média':<10} | {'Desvio-Padrão':<13} | {'Maior Valor':<12} | {'Menor Valor':<12}")
print("-" * 30 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 13)

estatisticas = {}

for i, (nome, chave) in enumerate(zip(nomes_modelos, chaves_modelos)):
    rss_valores = np.array(rss_resultados[chave])
    
    media = np.mean(rss_valores)
    desvio = np.std(rss_valores)
    maximo = np.max(rss_valores)
    minimo = np.min(rss_valores)
    
    estatisticas[chave] = {
        'media': media,
        'desvio': desvio,
        'maximo': maximo,
        'minimo': minimo
    }
    
    print(f"{nome:<30} | {media:<10.4f} | {desvio:<13.4f} | {maximo:<12.4f} | {minimo:<12.4f}")

print("="*80)

print(f"\n6. ANÁLISE DOS RESULTADOS")
print("-" * 50)


melhor_modelo = min(estatisticas.keys(), key=lambda k: estatisticas[k]['media'])
idx_melhor = chaves_modelos.index(melhor_modelo)
nome_melhor = nomes_modelos[idx_melhor]

print(f"MELHOR MODELO: {nome_melhor}")
print(f"RSS médio: {estatisticas[melhor_modelo]['media']:.4f}")


print(f"\nEFEITO DA REGULARIZAÇÃO (λ):")
lambdas = [0, 0.25, 0.5, 0.75, 1.0]
modelos_ridge = ['mqo_tradicional', 'ridge_025', 'ridge_050', 'ridge_075', 'ridge_100']
rss_medios_ridge = [estatisticas[modelo]['media'] for modelo in modelos_ridge]

for lambda_val, rss_medio in zip(lambdas, rss_medios_ridge):
    print(f"λ = {lambda_val:<4}: RSS médio = {rss_medio:.4f}")


print(f"\n7. GRÁFICOS DE ANÁLISE ESTATÍSTICA")
print("-" * 50)


plt.figure(3, figsize=(15, 10))

plt.subplot(2, 2, 1)
dados_boxplot = [rss_resultados[chave] for chave in chaves_modelos]
nomes_curtos = ['Média', 'MQO', 'R(0.25)', 'R(0.5)', 'R(0.75)', 'R(1.0)']
cores_box = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
box_plot = plt.boxplot(dados_boxplot, labels=nomes_curtos, patch_artist=True)
for patch, cor in zip(box_plot['boxes'], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
plt.title('Distribuição dos RSS por Modelo')
plt.ylabel('RSS')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 2)
plt.plot(lambdas, rss_medios_ridge, 'bo-', linewidth=3, markersize=10)
plt.title('Efeito da Regularização Ridge')
plt.xlabel('Parâmetro λ')
plt.ylabel('RSS Médio')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 3)
medias_rss = [estatisticas[chave]['media'] for chave in chaves_modelos]
cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
bars = plt.bar(nomes_curtos, medias_rss, color=cores, alpha=0.8)
plt.title('RSS Médio por Modelo')
plt.ylabel('RSS Médio')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)


for bar, valor in zip(bars, medias_rss):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(medias_rss), 
             f'{valor:.1f}', ha='center', va='bottom', fontsize=9)


plt.subplot(2, 2, 4)
desvios_rss = [estatisticas[chave]['desvio'] for chave in chaves_modelos]
bars = plt.bar(nomes_curtos, desvios_rss, color=cores, alpha=0.8)
plt.title('Desvio Padrão do RSS por Modelo')
plt.ylabel('Desvio Padrão RSS')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)


for bar, valor in zip(bars, desvios_rss):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(desvios_rss), 
             f'{valor:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

plt.show()

print("\n Análise concluída! Verifique os 3 gráficos:")
print(" Figure 1: Dados originais")
print(" Figure 2: COMPARAÇÃO DE TODOS OS MODELOS (cores diferentes)")
print(" Figure 3: Análises estatísticas")
print("="*80)