# Requirements

## REQ-01: Especificações da Atividade 01 - Perceptron
- **Descrição:** Implementação e treinamento de um Perceptron com 3 entradas para classificação em 2 classes (+1, -1).
- **Restrição:** Proibido o uso de bibliotecas prontas que já possuam o modelo implementado (ex: sklearn). Permitido apenas numpy.
- **Critérios de Aceite:**
  - O algoritmo usa a regra de Hebb supervisionada com taxa de aprendizagem de 0.01.
  - O treinamento é executado 5 vezes independentes, com pesos iniciais aleatórios (entre 0 e 1).
  - Tabela 1 é gerada com os pesos iniciais, finais e número de épocas de cada treinamento.
  - Tabela 2 é gerada aplicando os 5 modelos treinados sobre 10 amostras de teste.
  - O relatório final responde por que o número de épocas varia e qual a principal limitação do perceptron.
