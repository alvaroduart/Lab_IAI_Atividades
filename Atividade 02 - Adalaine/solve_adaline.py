import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import nbformat as nbf
import matplotlib.pyplot as plt

def extract_tables_from_docx(docx_path):
    with zipfile.ZipFile(docx_path) as docx:
        tree = ET.XML(docx.read('word/document.xml'))
        tables = []
        for table in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'):
            t = []
            for row in table.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
                r = []
                for cell in row.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
                    texts = [node.text for p in cell.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p') for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                    r.append(''.join(texts).strip())
                t.append(r)
            tables.append(t)
        return tables

tables = extract_tables_from_docx('context/Adaline.docx')

# Parse training data (Table 3)
train_data = []
for row in tables[3][1:]:
    try:
        train_data.append([float(x) for x in row[1:]])
    except ValueError:
        pass
train_data = np.array(train_data)
X_train = train_data[:, :-1]
d_train = train_data[:, -1]

# Add bias (-1)
# "x0 = -1" according to the image description
X_train_bias = np.insert(X_train, 0, -1.0, axis=1)

# Parse test data (Table 2)
test_data = []
for row in tables[2][1:]:
    try:
        test_data.append([float(x) for x in row[1:5]])
    except ValueError:
        pass
X_test = np.array(test_data)
X_test_bias = np.insert(X_test, 0, -1.0, axis=1)

# Adaline parameters
eta = 0.0025
precision = 1e-6
np.random.seed(42)

results = []
test_predictions = []
eqm_history_plots = []

for i in range(5):
    # Initialize weights between 0 and 1
    w_init = np.random.rand(5)
    w = w_init.copy()
    
    eqm_history = []
    epoch = 0
    
    while True:
        # Calculate u
        u = np.dot(X_train_bias, w)
        # Calculate eqm
        error = d_train - u
        eqm = np.mean(error ** 2)
        eqm_history.append(eqm)
        
        # Update weights (Delta rule)
        for j in range(len(X_train_bias)):
            uj = np.dot(X_train_bias[j], w)
            ej = d_train[j] - uj
            w = w + eta * ej * X_train_bias[j]
        
        if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
            break
        epoch += 1
        
    if i < 2:
        eqm_history_plots.append(eqm_history)
        
    results.append({
        'w_init': w_init,
        'w_final': w,
        'epochs': epoch
    })
    
    # Test predictions
    u_test = np.dot(X_test_bias, w)
    y_test = np.where(u_test >= 0, 1.0, -1.0)
    test_predictions.append(y_test)

# Format the results into README.md
readme_content = """# Centro Federal de Educação Tecnológica de Minas Gerais
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
"""

for i, res in enumerate(results):
    w_init_str = " ".join([f"`{x:.4f}`" for x in res['w_init']])
    w_final_str = " ".join([f"`{x:.4f}`" for x in res['w_final']])
    readme_content += f"| **{i+1}º (T{i+1})** | {w_init_str} | {w_final_str} | {res['epochs']} |\n"

readme_content += """
### 2. Para os dois primeiros treinamentos realizados acima trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento.

Os gráficos estão disponíveis no notebook `Adaline.ipynb`.

### 3. Para todos os treinamentos realizados acima, aplique a rede ADALINE para classificar e indicar ao comutador se os sinais abaixo devem ser encaminhados para a válvula A (-1) ou B (1).

| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

test_preds = np.array(test_predictions).T

for i, row in enumerate(X_test):
    row_str = " | ".join([f"{x:.4f}" for x in row])
    preds_str = " | ".join([f"{p:.1f}" for p in test_preds[i]])
    readme_content += f"| **{i+1}** | {row_str} | {preds_str} |\n"

readme_content += """
### 4. Embora o número de épocas de cada treinamento realizado no item 2 seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.

Ao contrário do Perceptron Simples, a rede ADALINE utiliza uma função de ativação linear para o cálculo do erro e a regra de aprendizado (Regra Delta) é baseada na minimização do Erro Quadrático Médio (EQM). A superfície de erro gerada pelo EQM em relação aos pesos do ADALINE forma um hiperparaboloide, possuindo um único mínimo global (o erro ótimo). O algoritmo converge (desce o gradiente) até encontrar este ponto ótimo independente de onde seja iniciado. 

Portanto, como o ponto ótimo é único para um dado conjunto de treinamento, os pesos finais sempre irão convergir para os mesmos valores (ou valores muito próximos, dependendo da precisão do critério de parada). O que muda de fato é a distância entre a inicialização aleatória dos pesos e o mínimo global da superfície do erro, fazendo com que o algoritmo necessite de mais ou menos iterações (épocas) para convergir, justificando a variação no número de épocas, mesmo atingindo os mesmos pesos.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Now generate Notebook
nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# ADALINE - Atividade 02\n\nImplementação do algoritmo de treinamento Regra Delta para a rede ADALINE e classificação das válvulas."),
    nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd"""),
    nbf.v4.new_markdown_cell("### Dados de Treinamento e Teste"),
    nbf.v4.new_code_cell(f"""# Conjunto de treinamento
X_train = np.array({X_train.tolist()})
d_train = np.array({d_train.tolist()})

# Adicionando o bias (x0 = -1)
X_train_bias = np.insert(X_train, 0, -1.0, axis=1)

# Conjunto de teste
X_test = np.array({X_test.tolist()})
X_test_bias = np.insert(X_test, 0, -1.0, axis=1)"""),
    nbf.v4.new_markdown_cell("### Algoritmo de Treinamento (Regra Delta)"),
    nbf.v4.new_code_cell("""eta = 0.0025
precision = 1e-6
np.random.seed(42)

results = []
eqm_history_plots = []
test_predictions = []

for i in range(5):
    # Inicializando pesos (w0 a w4) entre 0 e 1
    w = np.random.rand(5)
    w_init = w.copy()
    
    eqm_history = []
    epoch = 0
    
    while True:
        # Calcular EQM atual
        u = np.dot(X_train_bias, w)
        error = d_train - u
        eqm = np.mean(error ** 2)
        eqm_history.append(eqm)
        
        # Regra delta
        for j in range(len(X_train_bias)):
            uj = np.dot(X_train_bias[j], w)
            ej = d_train[j] - uj
            w = w + eta * ej * X_train_bias[j]
        
        if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
            break
        epoch += 1
        
    results.append({
        'Treinamento': f"T{i+1}",
        'Épocas': epoch,
        'Pesos Iniciais': np.round(w_init, 4),
        'Pesos Finais': np.round(w, 4)
    })
    
    if i < 2:
        eqm_history_plots.append(eqm_history)
        
    u_test = np.dot(X_test_bias, w)
    y_test = np.where(u_test >= 0, 1.0, -1.0)
    test_predictions.append(y_test)
"""),
    nbf.v4.new_markdown_cell("### Resultados do Treinamento"),
    nbf.v4.new_code_cell("""for res in results:
    print(f"Treinamento {res['Treinamento']}: {res['Épocas']} épocas")
    print(f"   Pesos Iniciais: {res['Pesos Iniciais']}")
    print(f"   Pesos Finais:   {res['Pesos Finais']}\\n")"""),
    nbf.v4.new_markdown_cell("### Gráficos do EQM em Função da Época"),
    nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
plt.plot(eqm_history_plots[0], label="Treinamento T1", color="blue")
plt.plot(eqm_history_plots[1], label="Treinamento T2", color="orange")
plt.title("Erro Quadrático Médio (EQM) por Época")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.legend()
plt.grid(True)
plt.show()"""),
    nbf.v4.new_markdown_cell("### Classificação das Amostras de Teste"),
    nbf.v4.new_code_cell("""test_preds = np.array(test_predictions).T

print("Amostra | y(T1) | y(T2) | y(T3) | y(T4) | y(T5)")
print("-" * 50)
for i in range(len(X_test)):
    preds_str = " | ".join([f"{p:4.1f}" for p in test_preds[i]])
    print(f"{i+1:7d} | {preds_str}")""")
]

with open('Adaline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
