import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def predicao(X,beta):
    return X@beta

data = np.loadtxt("Solubilidade.csv",delimiter=',')
controle_figura = True


fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2],
           edgecolor='k')
ax.set_title("Todo o conjunto de dados.")
ax.set_xlabel("Quantidade de Carbono")
ax.set_ylabel("Peso Molecular")
ax.set_zlabel("Solubilidade")



#Dados sem a coluna de 1s (X_)
X_ = data[:,:-1]
N,p = X_.shape
y = data[:,-1:]

#Dados com a coluna de 1s (X)
X = np.hstack((
    np.ones((N,1)),X_
))

rodadas = 1000

for r in range(rodadas):
    #EMBARALHAR O CONJUNTO DE DADOS
    idx = np.random.permutation(N)
    Xr_ = X_[idx,:]
    Xr = X[idx,:]
    yr = y[idx,:]

    #Particionamento do conjunto de dados (80/20)
    X_treino = Xr[:int(N*.8),:]
    X_treino_ = Xr_[:int(N*.8),:]
    y_treino = yr[:int(N*.8),:]

    X_teste = Xr[int(N*.8):,:]
    X_teste_ = Xr_[int(N*.8):,:]
    y_teste = yr[int(N*.8):,:]

    #Treinamento dos modelos:
    #Modelo baseado na média:
    beta_hat_media = np.array([
        [np.mean(y_treino)],
        [0],
        [0],
    ])

    #Modelo baseado MQO (sem intercepto)
    beta_hat_ = np.linalg.pinv(X_treino_.T@X_treino_)@X_treino_.T@y_treino
    beta_hat_ = np.vstack((
        np.zeros((1,1)),beta_hat_
    ))

    #Modelo MQO tradicional
    beta_hat = pinv(X_treino.T@X_treino)@X_treino.T@y_treino    

    '''
    Do ponto de vista prático, as linhas contidas neste if,
    servem para fins didáticos. Logo, em termos de fazer a estratégia de 
    validação por amostragem aleatória (várias rodadas de treino/teste)
    essas linhas não devem acontecem sempre, pois, o custo aumenta consideravelmente.
    Então, já fiz esse if ser executado apenas uma vez (na primeira rodada)


    '''
    if controle_figura:
        fig = plt.figure(2)
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.scatter(X_treino[:,1],X_treino[:,2],y_treino[:,0],edgecolor='k')
        ax.set_xlabel("Quantidade de Carbono")
        ax.set_ylabel("Peso Molecular")
        ax.set_zlabel("Solubilidade")
        ax.set_title("Observações de Treino ")
        
        x_axis = np.linspace(np.min(X_treino[:,1]),np.max(X_treino[:,1]))
        y_axis = np.linspace(np.min(X_treino[:,2]),np.max(X_treino[:,2]))
        X3d,Y3D = np.meshgrid(x_axis,y_axis)
        X_plot = np.concatenate((
            np.ones((X3d.shape[0],X3d.shape[1],1)),
            X3d.reshape(50,50,1),
            Y3D.reshape(50,50,1)
        ),axis=2)
        Z_plot = predicao(X_plot,beta_hat).reshape(50,50)
        ax.plot_surface(X3d,Y3D,Z_plot,edgecolor='k',
                        cmap='gray',rstride=10,cstride=10,alpha=.3,label='Modelo MQO')
       

        Z_plot = predicao(X_plot,beta_hat_).reshape(50,50)
        ax.plot_surface(X3d,Y3D,Z_plot,edgecolor='k',
                        cmap='jet',rstride=10,cstride=10,alpha=.3,label='MQO (sem intercepto)')

        Z_plot = predicao(X_plot,beta_hat_media).reshape(50,50)
        ax.plot_surface(X3d,Y3D,Z_plot,edgecolor='k',
                        cmap='turbo',rstride=10,cstride=10,alpha=.3,label= "Modelo baseado na média.")
        ax.legend()
        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.scatter(X_teste[:,1],X_teste[:,2],y_teste[:,0],edgecolor='k')
        ax.set_xlabel("Quantidade de Carbono")
        ax.set_title("Observações de Teste (desconhecidas)")
        ax.set_ylabel("Peso Molecular")
        ax.set_zlabel("Solubilidade")

        #Cálculo de epsilon (para comprovar as suposições do MQO)
        y_pred = predicao(X_treino,beta_hat)
        desvios = y_pred - y_treino

        plt.figure(3)
        plt.hist(desvios,edgecolor='k',color='green',bins=20)
        plt.title(r"Histograma de desvios ($\varepsilon$)")
        controle_figura = False

    #Teste de desempenho:
    #Modelo MQO (com intercepto)
    y_pred = predicao(X_teste,beta_hat)
    desvios = y_teste - y_pred
    SSE = np.sum(desvios**2)
    MSE = np.mean(desvios**2)
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    print(R2)
    bp=1
    y_pred = predicao(X_teste,beta_hat_)
    desvios = y_teste - y_pred
    SSE = np.sum(desvios**2)
    MSE = np.mean(desvios**2)
    SST = np.sum((y_teste - np.mean(y_teste))**2)
    R2 = 1 - SSE/SST
    print(R2)
    #Modelo média: Análise feita para dados de treinamento.
    #Essa análise deve ser modificada para os dados de teste.
    y_pred = predicao(X_treino,beta_hat_media)
    desvios = y_treino - y_pred
    SSE = np.sum(desvios**2)
    MSE = np.mean(desvios**2)
    SST = np.sum((y_treino - np.mean(y_treino))**2)
    R2 = 1 - SSE/SST
    print(R2)
    bp=1
    #Lembrete, para auxiliar vocês no trabalho, ainda falta a implementação
    #das variáveis que armazenam os desempenhos.
    #No final devem ser calculadas métricas estatísticas das 
    #séries (MSE,SSE e R2)  armazenadas







plt.show()  








bp = 1