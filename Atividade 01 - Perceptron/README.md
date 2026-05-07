# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 30/04/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes
**Valor:** 100% | **Nota:** _______

---

A partir da análise de um processo de destilação fracionada de petróleo observou-se que determinado óleo poderia ser classificado em duas classes de pureza {C1 e C2} a partir da medição de três grandezas {x1, x2 e x3} que representam algumas das propriedades físico-químicas do óleo. A equipe de engenheiros e cientistas pretende utilizar um perceptron para executar a classificação automática destas duas classes. 

Assim, baseado nas informações coletadas do processo formou-se o conjunto de treinamento em anexo, tomando por convenção o valor –1 para óleo pertencente à classe C1 e o valor +1 para óleo pertencente à classe C2. 

Portanto, o neurônio constituinte do perceptron terá três entradas e uma saída. Utilizando o algoritmo supervisionado de Hebb (regra de Hebb) para classificação de padrões e assumindo a taxa de aprendizagem igual a 0.01, faça as seguintes atividades:

### 1. Execute 5 treinamentos para a rede perceptron inicializando o vetor de pesos em cada treinamento com valores aleatórios entre zero e um. Se for o caso, reinicie o gerador de números aleatórios em cada treinamento de tal forma que os elementos do vetor de pesos iniciais não sejam os mesmos. Registre os resultados dos 5 treinamentos na tabela abaixo:

| Treinamento | Vetor de Pesos Inicial <br> w0 &nbsp;&nbsp;&nbsp; w1 &nbsp;&nbsp;&nbsp; w2 &nbsp;&nbsp;&nbsp; w3 | Vetor de Pesos Final <br> w0 &nbsp;&nbsp;&nbsp; w1 &nbsp;&nbsp;&nbsp; w2 &nbsp;&nbsp;&nbsp; w3 | Número de Épocas |
|:---:|:---|:---|:---:|
| **1º (T1)** | `0.5987` `0.3745` `0.9507` `0.7320` | `-2.8813` `1.4195` `2.3965` `-0.6730` | 352 |
| **2º (T2)** | `0.2406` `0.1151` `0.6091` `0.1334` | `-3.1194` `1.5979` `2.5179` `-0.7440` | 426 |
| **3º (T3)** | `0.3605` `0.8348` `0.1048` `0.7446` | `-2.8995` `1.4012` `2.4086` `-0.6944` | 338 |
| **4º (T4)** | `0.0773` `0.9890` `0.5495` `0.2814` | `-2.9227` `1.4341` `2.4044` `-0.6813` | 334 |
| **5º (T5)** | `0.7581` `0.7838` `0.6348` `0.2490` | `-3.0619` `1.5681` `2.4639` `-0.7313` | 408 |

### 2. Após o treinamento do perceptron aplique o mesmo na classificação automática das seguintes amostras de óleo, indicando na tabela abaixo os resultados das saídas (Classes) referentes aos cinco processos de treinamento realizados no item 1.

| Amostra | x1 | x2 | x3 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | -0.3565 | 0.0620 | 5.9891 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| **10** | -1.8842 | -0.2805 | 1.2548 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |

### 3. Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.

O algoritmo de aprendizado do perceptron ajusta os pesos iterativamente até que todas as amostras de treinamento sejam classificadas corretamente. Como os pesos iniciais em cada treinamento (T1 a T5) são gerados aleatoriamente, a inclinação e a posição iniciais do hiperplano de separação diferem em cada execução. Consequentemente, o número de ajustes iterativos necessários (épocas) para posicionar o hiperplano no local ideal de separação também varia. Quanto mais distante a inicialização aleatória dos pesos se encontrar do plano ótimo de separação, mais épocas serão necessárias para o algoritmo convergir.

### 4. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões.

A principal limitação do Perceptron Simples de camada única é que ele só consegue resolver problemas que são **linearmente separáveis**. Isso significa que ele apenas pode aprender e separar classes cujos dados correspondam a fronteiras de separação que podem ser traçadas de forma perfeitamente reta (um hiperplano). Se as classes tiverem sobreposições complexas ou dependerem de lógicas não-lineares, o perceptron falhará e ficará oscilando os pesos num loop infinito sem nunca convergir perfeitamente.

---

### ANEXO – Conjunto de Treinamento

| Padrão | x1 | x2 | x3 | d |
|:---:|:---:|:---:|:---:|:---:|
| 01 | -0.6508 | 0.1097 | 4.0009 | -1.0000 |
| 02 | -1.4492 | 0.8896 | 4.4005 | -1.0000 |
| 03 | 2.0850 | 0.6876 | 12.0710 | -1.0000 |
| 04 | 0.2626 | 1.1476 | 7.7985 | 1.0000 |
| 05 | 0.6418 | 1.0234 | 7.0427 | 1.0000 |
| 06 | 0.2569 | 0.6730 | 8.3265 | -1.0000 |
| 07 | 1.1155 | 0.6043 | 7.4446 | 1.0000 |
| 08 | 0.0914 | 0.3399 | 7.0677 | -1.0000 |
| 09 | 0.0121 | 0.5256 | 4.6316 | 1.0000 |
| 10 | -0.0429 | 0.4660 | 5.4323 | 1.0000 |
| 11 | 0.4340 | 0.6870 | 8.2287 | -1.0000 |
| 12 | 0.2735 | 1.0287 | 7.1934 | 1.0000 |
| 13 | 0.4839 | 0.4851 | 7.4850 | -1.0000 |
| 14 | 0.4089 | -0.1267 | 5.5019 | -1.0000 |
| 15 | 1.4391 | 0.1614 | 8.5843 | -1.0000 |
| 16 | -0.9115 | -0.1973 | 2.1962 | -1.0000 |
| 17 | 0.3654 | 1.0475 | 7.4858 | 1.0000 |
| 18 | 0.2144 | 0.7515 | 7.1699 | 1.0000 |
| 19 | 0.2013 | 1.0014 | 6.5489 | 1.0000 |
| 20 | 0.6483 | 0.2183 | 5.8991 | 1.0000 |
| 21 | -0.1147 | 0.2242 | 7.2435 | -1.0000 |
| 22 | -0.7970 | 0.8795 | 3.8762 | 1.0000 |
| 23 | -1.0625 | 0.6366 | 2.4707 | 1.0000 |
| 24 | 0.5307 | 0.1285 | 5.6883 | 1.0000 |
| 25 | -1.2200 | 0.7777 | 1.7252 | 1.0000 |
| 26 | 0.3957 | 0.1076 | 5.6623 | -1.0000 |
| 27 | -0.1013 | 0.5989 | 7.1812 | -1.0000 |
| 28 | 2.4482 | 0.9455 | 11.2095 | 1.0000 |
| 29 | 2.0149 | 0.6192 | 10.9263 | -1.0000 |
| 30 | 0.2012 | 0.2611 | 5.4631 | 1.0000 |
