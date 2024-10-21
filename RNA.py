import numpy as np

# Função de ativação sigmóide


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmóide


def sigmoid_derivada(x):
    return x * (1 - x)


# Dados de entrada
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Saídas esperadas
y = np.array([[0],
              [1],
              [1],
              [0]])

# Taxa de aprendizado
taxa_aprendizado = 0.1

# Inicialização dos pesos de forma aleatória
pesos_entrada_oculta = 2 * np.random.random((3, 4)) - 1
pesos_oculta_saida = 2 * np.random.random((4, 1)) - 1

# Treinamento da rede neural
for i in range(10000):

    # Forward propagation
    camada_entrada = X
    camada_oculta = sigmoid(np.dot(camada_entrada, pesos_entrada_oculta))
    camada_saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida))

    # Backpropagation
    erro_camada_saida = y - camada_saida
    delta_saida = erro_camada_saida * sigmoid_derivada(camada_saida)

    erro_camada_oculta = delta_saida.dot(pesos_oculta_saida.T)
    delta_oculta = erro_camada_oculta * sigmoid_derivada(camada_oculta)

    # Atualização dos pesos
    pesos_oculta_saida += camada_oculta.T.dot(delta_saida) * taxa_aprendizado
    pesos_entrada_oculta += camada_entrada.T.dot(
        delta_oculta) * taxa_aprendizado

print("Saída após o treinamento:")
print(camada_saida)
