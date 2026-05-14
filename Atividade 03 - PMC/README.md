# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 13/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

Para a confecção de um sistema de ressonância magnética, observou-se que é de extrema importância para o bom desempenho do processador de imagens que a variável {y} que mede a energia absorvida do sistema possa ser estimada a partir da medição de três outras grandezas {x1, x2, x3}. Entretanto, em função da complexidade do sistema, sabe-se que este mapeamento é de difícil obtenção por técnicas convencionais.

Assim, a equipe pretende utilizar uma rede perceptron multicamadas como um aproximador universal de funções, com topologia 3-10-1.

Utilizando o algoritmo de aprendizagem backpropagation (Regra Delta Generalizada) e os dados de treinamento apresentados no Anexo, realize as seguintes atividades:

### 1. Execute 5 treinamentos para a rede PERCEPTRON
Inicializando as matrizes de pesos em cada treinamento com valores aleatórios entre 0 e 1. Utilize a função de ativação logística para todos os neurônios, taxa de aprendizado $\eta = 0.1$ e precisão $\epsilon = 10^{-6}$.


| Treinamento | Erro Quadrático Médio | Número de Épocas |
|:---:|:---:|:---:|
| **T1** | 0.001521 | 131 |
| **T2** | 0.001528 | 120 |
| **T3** | 0.001553 | 134 |
| **T4** | 0.001536 | 120 |
| **T5** | 0.001550 | 119 |

### 2. Para os dois treinamentos acima com maiores números de épocas, trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento.

![Gráfico EQM](Atividade%2003%20-%20PMC/grafico_eqm.png)

Os gráficos também estão disponíveis no notebook `PMC.ipynb`.

### 3. Baseado na tabela do item 2, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento.

A rede neural perceptron multicamadas possui uma superfície de erro altamente não-linear e não-convexa em relação aos seus pesos, o que significa que existem múltiplos mínimos locais, além do mínimo global. A inicialização aleatória dos pesos determina o ponto de partida do algoritmo de descida do gradiente (Backpropagation) nessa superfície.

Diferentes pontos de partida (pesos iniciais) fazem com que o algoritmo siga trajetórias diferentes de otimização. Algumas trajetórias podem levar a um mínimo local "pior" ou convergir mais rapidamente para uma região plana (plateau), resultando em um Erro Quadrático Médio final e número de épocas distintos para cada treinamento. Além disso, a distância entre o ponto inicial e o ponto de convergência afeta diretamente a quantidade de iterações (épocas) necessárias para atingir o critério de parada.

### 4. Para todos os treinamentos efetuados no item 2, faça a validação da rede aplicando o conjunto de teste.

Forneça para cada treinamento o erro relativo médio (%) entre os valores desejados e os valores fornecidos pela rede. Obtenha também a respectiva variância.

| Amostra | x1 | x2 | x3 | d | y rede (T1) | y rede (T2) | y rede (T3) | y rede (T4) | y rede (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.0611 | 0.2860 | 0.7464 | 0.4831 | 0.4926 | 0.4895 | 0.4924 | 0.4916 | 0.4929 |
| **2** | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.5983 | 0.5995 | 0.5988 | 0.5976 | 0.5997 |
| **3** | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5329 | 0.5325 | 0.5335 | 0.5329 | 0.5347 |
| **4** | 0.9430 | 0.4476 | 0.2648 | 0.6843 | 0.7142 | 0.7145 | 0.7139 | 0.7135 | 0.7150 |
| **5** | 0.1399 | 0.1610 | 0.2477 | 0.2872 | 0.2973 | 0.2960 | 0.2923 | 0.2951 | 0.2934 |
| **6** | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7612 | 0.7605 | 0.7614 | 0.7613 | 0.7615 |
| **7** | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5751 | 0.5745 | 0.5774 | 0.5755 | 0.5765 |
| **8** | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6845 | 0.6847 | 0.6877 | 0.6874 | 0.6874 |
| **9** | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5385 | 0.5396 | 0.5399 | 0.5383 | 0.5399 |
| **10** | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.6087 | 0.6076 | 0.6115 | 0.6102 | 0.6110 |
| **11** | 0.1903 | 0.6523 | 0.7820 | 0.6950 | 0.6962 | 0.6964 | 0.6987 | 0.6983 | 0.6984 |
| **12** | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.6816 | 0.6817 | 0.6817 | 0.6809 | 0.6826 |
| **13** | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.3067 | 0.3048 | 0.3001 | 0.3031 | 0.3030 |
| **14** | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7906 | 0.7901 | 0.7885 | 0.7898 | 0.7896 |
| **15** | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.4722 | 0.4691 | 0.4717 | 0.4710 | 0.4718 |
| **16** | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.8281 | 0.8274 | 0.8262 | 0.8267 | 0.8266 |
| **17** | 0.2225 | 0.9182 | 0.7820 | 0.7581 | 0.7862 | 0.7863 | 0.7862 | 0.7865 | 0.7858 |
| **18** | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.5967 | 0.5974 | 0.5978 | 0.5967 | 0.5983 |
| **19** | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.8065 | 0.8070 | 0.8048 | 0.8060 | 0.8061 |
| **20** | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.5024 | 0.5002 | 0.5033 | 0.5022 | 0.5029 |
| **Erro Relativo Médio (%)** | | | | | 1.9664 | 1.8357 | 1.8201 | 1.8629 | 1.9268 |
| **Variância (%)** | | | | | 2.0871 | 1.9788 | 1.9290 | 1.9774 | 1.8945 |

### 5. Baseado nas análises da tabela acima indique qual das configurações finais de treinamento {T1, T2, T3, T4 ou T5} seria a mais adequada para o sistema de ressonância magnética.

Baseado na tabela de validação com o conjunto de teste, a configuração mais adequada para o sistema de ressonância magnética é o **Treinamento T3**. Esta configuração apresentou o menor **Erro Relativo Médio (1.82%)** e uma variância associada de **1.93%**, o que indica a melhor capacidade de generalização da rede Perceptron Multicamadas para estimar a variável de energia absorvida $y$ a partir de amostras não vistas no treinamento.
