# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 07/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

Um sistema de gerenciamento automático de ajuste de duas válvulas situado a 500 metros de um processo industrial envia um sinal codificado constituído de quatro grandezas {x1, x2, x3 e x4} que são necessárias para o ajuste de cada uma das válvulas. A mesma via de comunicação é utilizada para acionamento de ambas válvulas, sendo que o comutador localizado próximo das válvulas deve decidir se o sinal é para a válvula A ou B.

Utilizando o algoritmo de treinamento da Regra Delta para classificação de padrões no ADALINE, realize as seguintes atividades:

### 1. Execute 5 treinamentos para a rede ADALINE inicializando o vetor de pesos em cada treinamento com valores aleatórios entre zero e um. Utilize taxa de aprendizado $\eta = 0.0025$ e precisão $\epsilon = 10^{-6}$.

| Treinamento | Vetor de Pesos Inicial <br> w0 &nbsp;&nbsp;&nbsp; w1 &nbsp;&nbsp;&nbsp; w2 &nbsp;&nbsp;&nbsp; w3 &nbsp;&nbsp;&nbsp; w4 | Vetor de Pesos Final <br> w0 &nbsp;&nbsp;&nbsp; w1 &nbsp;&nbsp;&nbsp; w2 &nbsp;&nbsp;&nbsp; w3 &nbsp;&nbsp;&nbsp; w4 | Número de Épocas |
|:---:|:---|:---|:---:|
| **1º (T1)** | `0.3745` `0.9507` `0.7320` `0.5987` `0.1560` | `-1.8132` `1.3129` `1.6424` `-0.4276` `-1.1779` | 888 |
| **2º (T2)** | `0.1560` `0.0581` `0.8662` `0.6011` `0.7081` | `-1.8132` `1.3129` `1.6424` `-0.4277` `-1.1778` | 891 |
| **3º (T3)** | `0.0206` `0.9699` `0.8324` `0.2123` `0.1818` | `-1.8132` `1.3129` `1.6424` `-0.4278` `-1.1778` | 848 |
| **4º (T4)** | `0.1834` `0.3042` `0.5248` `0.4319` `0.2912` | `-1.8132` `1.3129` `1.6423` `-0.4278` `-1.1778` | 885 |
| **5º (T5)** | `0.6119` `0.1395` `0.2921` `0.3664` `0.4561` | `-1.8133` `1.3129` `1.6424` `-0.4279` `-1.1778` | 916 |

### 2. Para os dois primeiros treinamentos realizados acima trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento.

Os gráficos estão disponíveis no notebook `Adaline.ipynb`.

### 3. Para todos os treinamentos realizados acima, aplique a rede ADALINE para classificar e indicar ao comutador se os sinais abaixo devem ser encaminhados para a válvula A (-1) ou B (1).

| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.9694 | 0.6909 | 0.4334 | 3.4965 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **2** | 0.5427 | 1.3832 | 0.6390 | 4.0352 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **3** | 0.6081 | -0.9196 | 0.5925 | 0.1016 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **4** | -0.1618 | 0.4694 | 0.2030 | 3.0117 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **5** | 0.1870 | -0.2578 | 0.6124 | 1.7749 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **6** | 0.4891 | -0.5276 | 0.4378 | 0.6439 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **7** | 0.3777 | 2.0149 | 0.7423 | 3.3932 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **8** | 1.1498 | -0.4067 | 0.2469 | 1.5866 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **9** | 0.9325 | 1.0950 | 1.0359 | 3.3591 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **10** | 0.5060 | 1.3317 | 0.9222 | 3.7174 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **11** | 0.0497 | -2.0656 | 0.6124 | -0.6585 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **12** | 0.4004 | 3.5369 | 0.9766 | 5.3532 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **13** | -0.1874 | 1.3343 | 0.5374 | 3.2189 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **14** | 0.5060 | 1.3317 | 0.9222 | 3.7174 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **15** | 1.6375 | -0.7911 | 0.7537 | 0.5515 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### 4. Embora o número de épocas de cada treinamento realizado no item 2 seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.

Ao contrário do Perceptron Simples, a rede ADALINE utiliza uma função de ativação linear para o cálculo do erro e a regra de aprendizado (Regra Delta) é baseada na minimização do Erro Quadrático Médio (EQM). A superfície de erro gerada pelo EQM em relação aos pesos do ADALINE forma um hiperparaboloide, possuindo um único mínimo global (o erro ótimo). O algoritmo converge (desce o gradiente) até encontrar este ponto ótimo independente de onde seja iniciado. 

Portanto, como o ponto ótimo é único para um dado conjunto de treinamento, os pesos finais sempre irão convergir para os mesmos valores (ou valores muito próximos, dependendo da precisão do critério de parada). O que muda de fato é a distância entre a inicialização aleatória dos pesos e o mínimo global da superfície do erro, fazendo com que o algoritmo necessite de mais ou menos iterações (épocas) para convergir, justificando a variação no número de épocas, mesmo atingindo os mesmos pesos.
