import numpy as np
import matplotlib.pyplot as plt
from typing import Optional # Verificar esse typing, tá sendo utilizado só no MQOT

# Função customizada para dividir os dados em treino e teste
def dividir_treino_teste(caracteristicas, rotulos, proporcao_treino=0.8):
    indices = np.arange(caracteristicas.shape[0])
    np.random.shuffle(indices)
    indice_fim_treino = int(proporcao_treino * len(indices))
    caracteristicas_treino = caracteristicas[indices[:indice_fim_treino]]
    caracteristicas_teste = caracteristicas[indices[indice_fim_treino:]]
    rotulos_treino = rotulos[indices[:indice_fim_treino]]
    rotulos_teste = rotulos[indices[indice_fim_treino:]]
    return caracteristicas_treino, caracteristicas_teste, rotulos_treino, rotulos_teste

# Carregar o conjunto de dados com numpy - Tava usando pandas antes
dados = np.loadtxt("EMGsDataset.csv", delimiter=",")

# Organizar os dados em características (X) e rótulos (y)
sensor_corrugador = dados[0, :]  # Sensor 1 - Corrugador do Supercílio
sensor_zigomatico = dados[1, :]  # Sensor 2 - Zigomático Maior
rotulos = dados[2, :].astype(int)  # Rótulos das classes

# Criar matriz de características (X)
caracteristicas = np.column_stack((sensor_corrugador, sensor_zigomatico)) # N x p

# Definir cores e nomes para as cinco classes
cores_classes = ['green', 'magenta', 'red', 'cyan', 'yellow']
nomes = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']
nomes_classes = [f"{classe}" for classe in nomes]

classes_unicas = np.unique(rotulos)

# Plotar o gráfico de dispersão
plt.figure(figsize=(8, 6))
for indice, classe in enumerate(classes_unicas):
    pontos_classe = caracteristicas[rotulos == classe]
    plt.scatter(pontos_classe[:, 0], pontos_classe[:, 1], 
                color=cores_classes[indice], label=nomes_classes[indice], alpha=0.8, marker='^', edgecolors='black')

plt.xlabel("Sensor Corrugador")
plt.ylabel("Sensor Zigomático")
plt.title("Gráfico de Dispersão das Classes")
plt.legend(title="Classes")
plt.show()

# Função para calcular a Distância de Mahalanobis Quadrática - Ajuda a escolher a menor distância
def distancia_mahalanobis(amostra, media_classe, covariancia_inversa):
    diferenca = amostra - media_classe
    return np.dot(np.dot(diferenca.T, covariancia_inversa), diferenca)

# Classe MQOT para classificação
class ClassificadorMQOT:
    def __init__(self, caracteristicas: np.ndarray, rotulos: np.ndarray) -> None:
        self.caracteristicas = caracteristicas
        self.rotulos = rotulos
        self.coeficientes: Optional[np.ndarray] = None
        self.classes = np.unique(rotulos)
    
    def treinar(self) -> None:
        rotulos_binarios = np.array([self.rotulos == classe for classe in self.classes], dtype=int).T
        self.coeficientes = np.linalg.pinv(self.caracteristicas.T @ self.caracteristicas) @ self.caracteristicas.T @ rotulos_binarios
    
    def prever(self, novas_caracteristicas: np.ndarray) -> np.ndarray:
        if self.coeficientes is None:
            self.treinar()
        previsoes = novas_caracteristicas @ self.coeficientes
        indice_classes = np.argmax(previsoes, axis=1)
        return self.classes[indice_classes]

# Classificador Naive Bayes Gaussiano
class ClassificadorBayesIngenuoGaussiano:
    def __init__(self):
        self.classes = None
        self.medias_classes = {}
        self.covariancias_classes = {}
        self.prioris_classes = {}
    
    def treinar(self, caracteristicas, rotulos):
        self.classes = np.unique(rotulos)
        for classe in self.classes:
            caracteristicas_classe = caracteristicas[rotulos == classe]
            self.medias_classes[classe] = np.mean(caracteristicas_classe, axis=0)
            variancias = np.var(caracteristicas_classe, axis=0)
            self.covariancias_classes[classe] = np.diag(variancias)
            self.prioris_classes[classe] = len(caracteristicas_classe) / len(rotulos)
    
    def prever(self, novas_caracteristicas):
        previsoes = []
        for amostra in novas_caracteristicas:
            probabilidades_classe = []
            for classe in self.classes:
                media_classe = self.medias_classes[classe]
                covariancia_classe = self.covariancias_classes[classe]
                probabilidade = distancia_mahalanobis(amostra, media_classe, np.linalg.pinv(covariancia_classe))
                probabilidades_classe.append(probabilidade)
            previsoes.append(self.classes[np.argmin(probabilidades_classe)])
        return np.array(previsoes)

# Classificador LDA - Covariâncias Iguais
class ClassificadorLDA:
    def __init__(self):
        self.classes = None
        self.medias_classes = {}
        self.covariancia_geral = None
        self.prioris_classes = {}
    
    def treinar(self, caracteristicas, rotulos):
        self.classes = np.unique(rotulos)
        self.covariancia_geral = np.cov(caracteristicas, rowvar=False)
        for classe in self.classes:
            caracteristicas_classe = caracteristicas[rotulos == classe]
            self.medias_classes[classe] = np.mean(caracteristicas_classe, axis=0)
            self.prioris_classes[classe] = len(caracteristicas_classe) / len(rotulos)
    
    def prever(self, novas_caracteristicas):
        covariancia_inversa = np.linalg.pinv(self.covariancia_geral)
        previsoes = []
        for amostra in novas_caracteristicas:
            distancias_classe = []
            for classe in self.classes:
                media_classe = self.medias_classes[classe]
                distancia = distancia_mahalanobis(amostra, media_classe, covariancia_inversa)
                distancias_classe.append(distancia)
            previsoes.append(self.classes[np.argmin(distancias_classe)])
        return np.array(previsoes)

# Função para calcular a matriz de covariância agregada
def calcular_covariancia_agregada(caracteristicas_treino, rotulos_treino, num_classes):
    covariancias = []
    for classe in range(1, num_classes + 1):
        dados_classe = caracteristicas_treino[rotulos_treino == classe]
        covariancias.append(np.cov(dados_classe, rowvar=False))
    return np.mean(covariancias, axis=0)

# Função para calcular as covariâncias regularizadas por Friedman
def calcular_covariancias_regularizadas_friedman(caracteristicas_treino, rotulos_treino, lambdas):
    num_classes = len(np.unique(rotulos_treino))
    num_amostras = len(caracteristicas_treino)
    covariancia_agregada = calcular_covariancia_agregada(caracteristicas_treino, rotulos_treino, num_classes)
    medias_classes = {}
    covariancias_regularizadas = {}
    inversas_covariancias = {}
    determinantes_covariancias = {}

    for classe in range(1, num_classes + 1):
        dados_classe = caracteristicas_treino[rotulos_treino == classe]
        num_amostras_classe = len(dados_classe)
        media_classe = np.mean(dados_classe, axis=0)
        covariancia_classe = np.cov(dados_classe, rowvar=False)

        medias_classes[classe] = media_classe
        covariancias_regularizadas[classe] = {}
        inversas_covariancias[classe] = {}
        determinantes_covariancias[classe] = {}

        for lam in lambdas:
            covariancia_reg = ((1 - lam) * num_amostras_classe * covariancia_classe + lam * num_amostras * covariancia_agregada) / ((1 - lam) * num_amostras_classe + lam * num_amostras)
            covariancias_regularizadas[classe][lam] = covariancia_reg
            inversas_covariancias[classe][lam] = np.linalg.pinv(covariancia_reg)
            determinantes_covariancias[classe][lam] = np.linalg.det(covariancia_reg)

    return medias_classes, covariancias_regularizadas, inversas_covariancias, determinantes_covariancias

# Função para prever a classe usando o Classificador Gaussiano com regularização por Friedman
def prever_gaussiano_friedman(amostra, medias_classes, inversas_covariancias, determinantes_covariancias, lambda_):
    probabilidades = []
    for classe in medias_classes:
        media_classe = medias_classes[classe]
        diferenca = amostra - media_classe
        cov_inversa = inversas_covariancias[classe][lambda_]
        det_cov = determinantes_covariancias[classe][lambda_]
        
        discriminante = -0.5 * (diferenca @ cov_inversa @ diferenca.T) - 0.5 * np.log(det_cov)
        probabilidades.append(discriminante)

    return np.argmax(probabilidades) + 1

# Parâmetros para validação Monte Carlo
num_rodadas = 500  # Número de rodadas para validação Monte Carlo - 10 pra teste, 500 para o trabalho
lambdas = [0, 0.25, 0.5, 0.75, 1] # Invés de fazer um para o Tradicional
acuracias_lda = []
acuracias_bayes = []
acuracias_mqot = []
acuracias_gaussiano_reg = {lmbd: [] for lmbd in lambdas}

# Loop para Monte Carlo com várias rodadas
for _ in range(num_rodadas):
    caracteristicas_treino, caracteristicas_teste, rotulos_treino, rotulos_teste = dividir_treino_teste(caracteristicas, rotulos, proporcao_treino=0.8)

    # Classificador Gaussiano (Covariâncias Iguais)
    lda = ClassificadorLDA()
    lda.treinar(caracteristicas_treino, rotulos_treino)
    previsoes = lda.prever(caracteristicas_teste)
    acuracia = np.mean(rotulos_teste == previsoes)
    acuracias_lda.append(acuracia)
    
    # Classificador Bayes Ingênuo Gaussiano
    bayes_gaussiano = ClassificadorBayesIngenuoGaussiano()
    bayes_gaussiano.treinar(caracteristicas_treino, rotulos_treino)
    previsoes = bayes_gaussiano.prever(caracteristicas_teste)
    acuracia = np.mean(rotulos_teste == previsoes)
    acuracias_bayes.append(acuracia)

    # Classificador MQOT
    mqot = ClassificadorMQOT(caracteristicas_treino, rotulos_treino)
    mqot.treinar()
    previsoes = mqot.prever(caracteristicas_teste)
    acuracia = np.mean(rotulos_teste == previsoes)
    acuracias_mqot.append(acuracia)

    # Classificador Gaussiano Regularizado Friedman
    medias_classes, covariancias_regularizadas, inversas_covariancias, determinantes_covariancias = calcular_covariancias_regularizadas_friedman(caracteristicas_treino, rotulos_treino, lambdas)
    for lam in lambdas:
        previsoes = [prever_gaussiano_friedman(amostra, medias_classes, inversas_covariancias, determinantes_covariancias, lam) for amostra in caracteristicas_teste]
        acuracia = np.mean(rotulos_teste == previsoes)
        acuracias_gaussiano_reg[lam].append(acuracia)

# Cálculo das estatísticas de acurácia para os classificadores
resultados = {
    'Modelo': [],
    'Acurácia Média': [],
    'Desvio Padrão': [],
    'Acurácia Máxima': [],
    'Acurácia Mínima': []
}

def adicionar_resultados(nome_modelo, acuracias):
    resultados['Modelo'].append(nome_modelo)
    resultados['Acurácia Média'].append(np.mean(acuracias))
    resultados['Desvio Padrão'].append(np.std(acuracias))
    resultados['Acurácia Máxima'].append(np.max(acuracias))
    resultados['Acurácia Mínima'].append(np.min(acuracias))

# Adicionar resultados de cada classificador
adicionar_resultados('MQOT', acuracias_mqot)
adicionar_resultados('Bayes Ingênuo', acuracias_bayes)
adicionar_resultados('Covariância Iguais', acuracias_lda)
for lam, acuracias in acuracias_gaussiano_reg.items():
    if lam == 1:
        adicionar_resultados('Covariância Agregada', acuracias)
    elif lam == 0:
        adicionar_resultados('Tradicional', acuracias)
    else:
        adicionar_resultados(f'Regularizado (λ={lam})', acuracias)

# Configuração da tabela para exibir com matplotlib
fig, ax = plt.subplots(figsize=(10, 6))  # Tamanho da tabela
ax.axis('off')  # Remove os eixos do gráfico

# Dados para a tabela
colunas = ['Modelo', 'Acurácia Média', 'Desvio Padrão', 'Acurácia Máxima', 'Acurácia Mínima']
linhas = [
    [resultados['Modelo'][i], 
     f"{resultados['Acurácia Média'][i]:.4f}", 
     f"{resultados['Desvio Padrão'][i]:.4f}", 
     f"{resultados['Acurácia Máxima'][i]:.4f}", 
     f"{resultados['Acurácia Mínima'][i]:.4f}"] 
    for i in range(len(resultados['Modelo']))
]

# Criação da tabela
tabela = ax.table(cellText=linhas, colLabels=colunas, cellLoc='center', loc='center')
tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.2)

plt.title("Resultados de Acurácia dos Modelos", fontsize=14)
plt.show()