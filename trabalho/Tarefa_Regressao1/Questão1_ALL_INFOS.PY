import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

data = np.loadtxt("trabalho/aerogerador.dat", delimiter=None)

X = data[:, 0].reshape(-1, 1)  # N x 1 matriz (coluna)
Y = data[:, 1].reshape(-1, 1)  # N x 1 vetor (coluna)

N,p = X.shape

# ====================================================================
# 1. VISUALIZAÇÃO INICIAL DOS DADOS (QUESTÃO 1)
# ====================================================================

# Dados originais para visualização (sem coluna de 1s)
X_original = data[:, 0]  # Velocidade do vento
Y_original = data[:, 1]  # Potência gerada

print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)
print(f"Número total de observações: {N}")
print(f"Velocidade do vento - Min: {np.min(X_original):.2f}, Max: {np.max(X_original):.2f}, Média: {np.mean(X_original):.2f}")
print(f"Potência gerada - Min: {np.min(Y_original):.2f}, Max: {np.max(Y_original):.2f}, Média: {np.mean(Y_original):.2f}")

# Gráfico de espalhamento
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_original, Y_original, alpha=0.6, color='blue', s=30)
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Espalhamento\n(Velocidade do Vento vs Potência)')
plt.grid(True, alpha=0.3)

# Adicionar linha de tendência para visualizar o padrão
z = np.polyfit(X_original, Y_original, 1)
p = np.poly1d(z)
plt.plot(X_original, p(X_original), "r--", alpha=0.8, label=f'Tendência Linear')
plt.legend()

# Histograma da variável independente
plt.subplot(2, 2, 2)
plt.hist(X_original, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Frequência')
plt.title('Distribuição da Velocidade do Vento')
plt.grid(True, alpha=0.3)

# Histograma da variável dependente
plt.subplot(2, 2, 3)
plt.hist(Y_original, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Potência Gerada')
plt.ylabel('Frequência')
plt.title('Distribuição da Potência Gerada')
plt.grid(True, alpha=0.3)

# Estatísticas descritivas
plt.subplot(2, 2, 4)
correlacao = np.corrcoef(X_original, Y_original)[0, 1]

stats_text = f"""
Estatísticas dos Dados:

Velocidade do Vento:
• Média: {np.mean(X_original):.2f}
• Desvio: {np.std(X_original):.2f}
• Range: {np.ptp(X_original):.2f}

Potência Gerada:
• Média: {np.mean(Y_original):.2f}
• Desvio: {np.std(Y_original):.2f}
• Range: {np.ptp(Y_original):.2f}

Correlação de Pearson: {correlacao:.4f}
"""

plt.text(0.5, 0.5, stats_text, ha='center', va='center', 
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
plt.axis('off')

plt.suptitle('ANÁLISE EXPLORATÓRIA DOS DADOS DO AEROGERADOR', fontsize=16, y=0.98)
plt.tight_layout()
plt.show(block=False)  # Não bloquear para permitir múltiplas janelas

#Dados com a coluna de 1s (X)
X = np.hstack((
    np.ones((N,1)),X
))

# ====================================================================
# SIMULAÇÃO MONTE CARLO PARA VALIDAÇÃO DOS MODELOS (R = 500 rodadas)
# ====================================================================

# Inicializar listas para armazenar RSS de cada modelo nas 500 rodadas
rss_media = []           # Modelo baseado na média
rss_mqo = []             # MQO tradicional
rss_ridge_025 = []       # MQO regularizado λ=0.25
rss_ridge_05 = []        # MQO regularizado λ=0.5
rss_ridge_075 = []       # MQO regularizado λ=0.75
rss_ridge_1 = []         # MQO regularizado λ=1

# Definir valores de lambda para regularização
lambda_values = [0.25, 0.5, 0.75, 1]
I = np.eye(2)  # Matriz identidade 2x2 (intercepto + 1 variável)

print("Iniciando simulação Monte Carlo com 500 rodadas...")
print("=" * 60)

# Loop principal: 500 rodadas de simulação Monte Carlo
for rodada in range(500):
    # ETAPA 1: REPARTICIONAMENTO DOS DADOS em cada rodada
    # Embaralhar os dados de forma diferente a cada rodada
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    Yr = Y[idx,:]

    # Dividir em 80% treino e 20% teste
    X_treino_mc = Xr[:int(N*0.8),:]
    Y_treino_mc = Yr[:int(N*0.8),:]
    X_teste_mc = Xr[int(N*0.8):,:]
    Y_teste_mc = Yr[int(N*0.8):,:]

    # ETAPA 2: TREINAR TODOS OS MODELOS

    # Modelo 1: Baseado na média (baseline)
    # β = [média_Y, 0] - ignora a variável X
    beta_media = np.array([[np.mean(Y_treino_mc)], [0]])

    # Modelo 2: MQO tradicional
    # β = (X'X)⁻¹X'Y
    beta_mqo = pinv(X_treino_mc.T @ X_treino_mc) @ X_treino_mc.T @ Y_treino_mc

    # Modelos 3-6: MQO regularizado para cada valor de λ
    betas_ridge = []
    for lam in lambda_values:
        # β = (X'X + λI)⁻¹X'Y
        beta_ridge = pinv(X_treino_mc.T @ X_treino_mc + lam * I) @ X_treino_mc.T @ Y_treino_mc
        betas_ridge.append(beta_ridge)

    # ETAPA 3: FAZER PREDIÇÕES NO CONJUNTO DE TESTE
    y_pred_media = X_teste_mc @ beta_media
    y_pred_mqo = X_teste_mc @ beta_mqo

    # ETAPA 4: CALCULAR RSS (Residual Sum of Squares) para cada modelo
    # RSS = Σ(y_real - y_predito)²

    # RSS do modelo de média
    rss_media.append(float(np.sum((Y_teste_mc - y_pred_media)**2)))

    # RSS do MQO tradicional
    rss_mqo.append(float(np.sum((Y_teste_mc - y_pred_mqo)**2)))

    # RSS dos modelos regularizados
    for i, beta_ridge in enumerate(betas_ridge):
        y_pred_ridge = X_teste_mc @ beta_ridge
        rss_val = float(np.sum((Y_teste_mc - y_pred_ridge)**2))

        # Armazenar RSS na lista correspondente ao λ
        if i == 0:    rss_ridge_025.append(rss_val)    # λ=0.25
        elif i == 1:  rss_ridge_05.append(rss_val)     # λ=0.5
        elif i == 2:  rss_ridge_075.append(rss_val)    # λ=0.75
        elif i == 3:  rss_ridge_1.append(rss_val)      # λ=1

print("Simulação Monte Carlo concluída!")
print("=" * 60)


# RESULTADOS: Calcular médias dos RSS ao longo das 500 rodadas
# ====================================================================
# ANÁLISE ESTATÍSTICA DOS RESULTADOS
# ====================================================================

# Organizar dados para análise
modelos = ['Media', 'MQO', 'Ridge 0.25', 'Ridge 0.5', 'Ridge 0.75', 'Ridge 1.0']
rss_listas = [rss_media, rss_mqo, rss_ridge_025, rss_ridge_05, rss_ridge_075, rss_ridge_1]

# Calcular estatísticas para cada modelo
estatisticas = []
for i, rss_lista in enumerate(rss_listas):
    stats = {
        'Modelo': modelos[i],
        'Media': np.mean(rss_lista),
        'Desvio_Padrao': np.std(rss_lista),
        'Minimo': np.min(rss_lista),
        'Maximo': np.max(rss_lista),
        'Mediana': np.median(rss_lista)
    }
    estatisticas.append(stats)

# TABELA DE RESULTADOS
print("\n" + "="*80)
print("TABELA DE ESTATÍSTICAS DOS RSS (500 RODADAS MONTE CARLO)")
print("="*80)
print(f"{'Modelo':<15} {'Media':<12} {'Desvio Padrao':<12} {'Minimo':<12} {'Maximo':<12} {'Mediana':<12}")
print("-" * 80)

for stats in estatisticas:
    print(f"{stats['Modelo']:<15} {stats['Media']:<12.2f} {stats['Desvio_Padrao']:<12.2f} "
          f"{stats['Minimo']:<12.2f} {stats['Maximo']:<12.2f} {stats['Mediana']:<12.2f}")

print("="*80)

# RANKING DE PERFORMANCE (menor RSS = melhor)
ranking = sorted(estatisticas, key=lambda x: x['Media'])
print(f"\nRANKING DE PERFORMANCE (RSS medio - menor e melhor):")
print("-" * 50)
for i, modelo in enumerate(ranking, 1):
    print(f"{i}o lugar: {modelo['Modelo']} - RSS medio: {modelo['Media']:.2f}")

# ANÁLISE DE VARIABILIDADE (menor desvio = mais estável)
ranking_estabilidade = sorted(estatisticas, key=lambda x: x['Desvio_Padrao'])
print(f"\nRANKING DE ESTABILIDADE (menor desvio padrao = mais estavel):")
print("-" * 60)
for i, modelo in enumerate(ranking_estabilidade, 1):
    print(f"{i}o lugar: {modelo['Modelo']} - Desvio: {modelo['Desvio_Padrao']:.2f}")

# GRÁFICOS DE ANÁLISE
print(f"\nGerando gráficos de análise...")

# Gráfico 1: Boxplot comparativo dos RSS
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.boxplot(rss_listas, labels=modelos)
plt.title('Distribuição dos RSS por Modelo\n(500 rodadas Monte Carlo)')
plt.ylabel('RSS')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 2: Barras com médias e desvios
plt.subplot(2, 2, 2)
medias = [stats['Media'] for stats in estatisticas]
desvios = [stats['Desvio_Padrao'] for stats in estatisticas]
x_pos = range(len(modelos))

plt.bar(x_pos, medias, yerr=desvios, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('RSS Medio +/- Desvio Padrao')
plt.ylabel('RSS')
plt.xticks(x_pos, modelos, rotation=45)
plt.grid(True, alpha=0.3)

# Gráfico 3: Histograma do melhor modelo (menor RSS médio)
melhor_modelo_idx = ranking[0]['Modelo']
melhor_rss = rss_listas[modelos.index(melhor_modelo_idx)]

plt.subplot(2, 2, 3)
plt.hist(melhor_rss, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title(f'Distribuição RSS - {melhor_modelo_idx}\n(Melhor modelo)')
plt.xlabel('RSS')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)

# Gráfico 4: Comparação RSS médio vs Desvio Padrão
plt.subplot(2, 2, 4)
cores = ['red', 'blue', 'orange', 'purple', 'brown', 'pink']
for i, stats in enumerate(estatisticas):
    plt.scatter(stats['Media'], stats['Desvio_Padrao'],
               s=100, c=cores[i], label=stats['Modelo'], alpha=0.7)

plt.xlabel('RSS Medio')
plt.ylabel('Desvio Padrao')
plt.title('Performance vs Estabilidade')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# DISCUSSÃO DOS RESULTADOS
print("\n" + "="*80)
print("DISCUSSÃO DOS RESULTADOS")
print("="*80)

print(f"\n1. PERFORMANCE GERAL:")
print(f"   • Melhor modelo: {ranking[0]['Modelo']} (RSS medio: {ranking[0]['Media']:.2f})")
print(f"   • Pior modelo: {ranking[-1]['Modelo']} (RSS medio: {ranking[-1]['Media']:.2f})")
print(f"   • Melhoria: {ranking[-1]['Media']/ranking[0]['Media']:.1f}x reducao no erro")

print(f"\n2. ESTABILIDADE:")
print(f"   • Mais estavel: {ranking_estabilidade[0]['Modelo']} (sigma: {ranking_estabilidade[0]['Desvio_Padrao']:.2f})")
print(f"   • Menos estavel: {ranking_estabilidade[-1]['Modelo']} (sigma: {ranking_estabilidade[-1]['Desvio_Padrao']:.2f})")

# Análise dos modelos regularizados
modelos_ridge = [stats for stats in estatisticas if 'Ridge' in stats['Modelo']]
if modelos_ridge:
    melhor_ridge = min(modelos_ridge, key=lambda x: x['Media'])
    print(f"\n3. REGULARIZACAO:")
    print(f"   • Melhor lambda: {melhor_ridge['Modelo']} (RSS: {melhor_ridge['Media']:.2f})")

    # Verificar se regularização ajudou
    mqo_tradicional = next(stats for stats in estatisticas if stats['Modelo'] == 'MQO')
    if melhor_ridge['Media'] < mqo_tradicional['Media']:
        print(f"   • Regularizacao MELHOROU o modelo (reducao de {mqo_tradicional['Media'] - melhor_ridge['Media']:.2f})")
    else:
        print(f"   • Regularizacao PIOROU o modelo (aumento de {melhor_ridge['Media'] - mqo_tradicional['Media']:.2f})")

print(f"\n4. CONCLUSOES:")
print(f"   • O modelo de media serve apenas como baseline (performance muito baixa)")
print(f"   • Modelos que consideram a velocidade do vento sao significativamente melhores")
print(f"   • A regularizacao pode ajudar ou prejudicar dependendo do valor de lambda")
print(f"   • Importante avaliar tanto performance quanto estabilidade na escolha do modelo")

print("="*80)