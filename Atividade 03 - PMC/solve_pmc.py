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

tables = extract_tables_from_docx('context/PMC1.docx')

# Parse training data (Table 3)
train_data = []
for row in tables[3][1:]:
    for i in range(3):
        try:
            if not row[i*5]: continue
            x1 = float(row[i*5+1])
            x2 = float(row[i*5+2])
            x3 = float(row[i*5+3])
            d = float(row[i*5+4])
            train_data.append([x1, x2, x3, d])
        except (ValueError, IndexError):
            pass

train_data = np.array(train_data)
X_train = train_data[:, :3]
d_train = train_data[:, 3]

# Add bias (-1)
X_train_bias = np.insert(X_train, 0, -1.0, axis=1)

# Parse test data (Table 2)
test_data = []
for row in tables[2][1:21]:
    try:
        x1 = float(row[1])
        x2 = float(row[2])
        x3 = float(row[3])
        d = float(row[4])
        test_data.append([x1, x2, x3, d])
    except ValueError:
        pass

test_data = np.array(test_data)
X_test = test_data[:, :3]
d_test = test_data[:, 3]
X_test_bias = np.insert(X_test, 0, -1.0, axis=1)

def sigmoid(x):
    # To prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

eta = 0.1
precision = 1e-6
np.random.seed(42)

results = []
eqm_history_plots = []
test_predictions = []

for i in range(5):
    # Random weights between 0 and 1
    # W1: 4x10, W2: 11x1
    W1 = np.random.rand(4, 10)
    W2 = np.random.rand(11, 1)
    
    eqm_history = []
    epoch = 0
    
    while True:
        # Forward pass on whole dataset to calculate EQM
        H_total = sigmoid(np.dot(X_train_bias, W1))
        H_total_bias = np.insert(H_total, 0, -1.0, axis=1)
        Y_total = sigmoid(np.dot(H_total_bias, W2))
        
        error_total = d_train.reshape(-1, 1) - Y_total
        eqm = np.mean(error_total ** 2)
        eqm_history.append(eqm)
        
        # Stop condition
        if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
            break
            
        # Update weights sample by sample (Stochastic/Online)
        for j in range(len(X_train_bias)):
            x_j = X_train_bias[j].reshape(1, 4)
            d_j = d_train[j]
            
            # Forward
            u_h = np.dot(x_j, W1)
            h_j = sigmoid(u_h)
            h_j_bias = np.insert(h_j, 0, -1.0, axis=1) # 1x11
            u_y = np.dot(h_j_bias, W2)
            y_j = sigmoid(u_y) # 1x1
            
            # Backprop
            error_y = d_j - y_j
            delta_out = error_y * sigmoid_derivative(y_j) # 1x1
            
            # W2 shape is 11x1. W2[1:] is 10x1. delta_out is 1x1.
            delta_h = sigmoid_derivative(h_j) * np.dot(delta_out, W2[1:].T) # 1x10
            
            # Update weights
            W2 += eta * np.dot(h_j_bias.T, delta_out) # 11x1
            W1 += eta * np.dot(x_j.T, delta_h) # 4x10
            
        epoch += 1

    results.append({
        'Treinamento': f"T{i+1}",
        'eqm': eqm_history[-1],
        'epochs': epoch,
        'W1_final': W1.copy(),
        'W2_final': W2.copy()
    })
    
    eqm_history_plots.append((f"T{i+1}", eqm_history))
    
    # Predict on test set
    H_test = sigmoid(np.dot(X_test_bias, W1))
    H_test_bias = np.insert(H_test, 0, -1.0, axis=1)
    Y_test = sigmoid(np.dot(H_test_bias, W2)).flatten()
    test_predictions.append(Y_test)

# Calculate relative error and variance
test_preds = np.array(test_predictions) # 5x20
erros_relativos = np.abs(test_preds - d_test) / d_test * 100
erro_relativo_medio = np.mean(erros_relativos, axis=1)
variancia = np.var(erros_relativos, axis=1)

# Format the results into README.md
readme_content = """# Centro Federal de Educação Tecnológica de Minas Gerais
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
Inicializando as matrizes de pesos em cada treinamento com valores aleatórios entre 0 e 1. Utilize a função de ativação logística para todos os neurônios, taxa de aprendizado $\\eta = 0.1$ e precisão $\\epsilon = 10^{-6}$.


| Treinamento | Erro Quadrático Médio | Número de Épocas |
|:---:|:---:|:---:|
"""

for res in results:
    readme_content += f"| **{res['Treinamento']}** | {res['eqm']:.6f} | {res['epochs']} |\n"

readme_content += """
### 2. Para os dois treinamentos acima com maiores números de épocas, trace os respectivos gráficos dos valores de erro quadrático médio (EQM) em função de cada época de treinamento.

![Gráfico EQM](grafico_eqm.png)

Os gráficos também estão disponíveis no notebook `PMC.ipynb`.

### 3. Baseado na tabela do item 2, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento.

A rede neural perceptron multicamadas possui uma superfície de erro altamente não-linear e não-convexa em relação aos seus pesos, o que significa que existem múltiplos mínimos locais, além do mínimo global. A inicialização aleatória dos pesos determina o ponto de partida do algoritmo de descida do gradiente (Backpropagation) nessa superfície.

Diferentes pontos de partida (pesos iniciais) fazem com que o algoritmo siga trajetórias diferentes de otimização. Algumas trajetórias podem levar a um mínimo local "pior" ou convergir mais rapidamente para uma região plana (plateau), resultando em um Erro Quadrático Médio final e número de épocas distintos para cada treinamento. Além disso, a distância entre o ponto inicial e o ponto de convergência afeta diretamente a quantidade de iterações (épocas) necessárias para atingir o critério de parada.

### 4. Para todos os treinamentos efetuados no item 2, faça a validação da rede aplicando o conjunto de teste.

Forneça para cada treinamento o erro relativo médio (%) entre os valores desejados e os valores fornecidos pela rede. Obtenha também a respectiva variância.

| Amostra | x1 | x2 | x3 | d | y rede (T1) | y rede (T2) | y rede (T3) | y rede (T4) | y rede (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for i in range(len(X_test)):
    row_str = f"{X_test[i][0]:.4f} | {X_test[i][1]:.4f} | {X_test[i][2]:.4f} | {d_test[i]:.4f}"
    preds_str = " | ".join([f"{test_preds[t][i]:.4f}" for t in range(5)])
    readme_content += f"| **{i+1}** | {row_str} | {preds_str} |\n"

readme_content += "| **Erro Relativo Médio (%)** | | | | | "
readme_content += " | ".join([f"{e:.4f}" for e in erro_relativo_medio]) + " |\n"
readme_content += "| **Variância (%)** | | | | | "
readme_content += " | ".join([f"{v:.4f}" for v in variancia]) + " |\n\n"

# Find best generalizing model (lowest relative mean error on test set)
best_model_idx = np.argmin(erro_relativo_medio)

readme_content += f"""### 5. Baseado nas análises da tabela acima indique qual das configurações finais de treinamento {{T1, T2, T3, T4 ou T5}} seria a mais adequada para o sistema de ressonância magnética.

Baseado na tabela de validação com o conjunto de teste, a configuração mais adequada para o sistema de ressonância magnética é o **Treinamento T{best_model_idx+1}**. Esta configuração apresentou o menor **Erro Relativo Médio ({erro_relativo_medio[best_model_idx]:.2f}%)** e uma variância associada de **{variancia[best_model_idx]:.2f}%**, o que indica a melhor capacidade de generalização da rede Perceptron Multicamadas para estimar a variável de energia absorvida $y$ a partir de amostras não vistas no treinamento.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Now generate Notebook
nb = nbf.v4.new_notebook()

# Find the two trainings with max epochs
top_epochs_idx = np.argsort([-res['epochs'] for res in results])[:2]
plot1_data = eqm_history_plots[top_epochs_idx[0]]
plot2_data = eqm_history_plots[top_epochs_idx[1]]

nb.cells = [
    nbf.v4.new_markdown_cell("# Perceptron Multicamadas (PMC) - Atividade 03\n\nImplementação do algoritmo backpropagation para aproximação de funções (sistema de ressonância magnética)."),
    nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt"""),
    nbf.v4.new_markdown_cell("### Dados de Treinamento e Teste"),
    nbf.v4.new_code_cell(f"""# Conjunto de treinamento
X_train = np.array({X_train.tolist()})
d_train = np.array({d_train.tolist()})

# Adicionando o bias (x0 = -1)
X_train_bias = np.insert(X_train, 0, -1.0, axis=1)

# Conjunto de teste
X_test = np.array({X_test.tolist()})
X_test_bias = np.insert(X_test, 0, -1.0, axis=1)"""),
    nbf.v4.new_markdown_cell("### Algoritmo de Treinamento (Backpropagation)"),
    nbf.v4.new_code_cell("""def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

eta = 0.1
precision = 1e-6
np.random.seed(42)

# O código completo de treinamento com as 5 inicializações diferentes encontra-se no script auxiliar e seus resultados estão sumarizados abaixo.
"""),
    nbf.v4.new_markdown_cell("### Gráficos do EQM em Função da Época para os Treinamentos com Mais Épocas"),
    nbf.v4.new_code_cell(f"""eqm_plot1 = {plot1_data[1]}
eqm_plot2 = {plot2_data[1]}

plt.figure(figsize=(12, 5))

# Gráfico 1
plt.subplot(1, 2, 1)
plt.plot(eqm_plot1, label="Treinamento {plot1_data[0]}", color="blue")
plt.title(f"Erro Quadrático Médio (EQM) - Treinamento {plot1_data[0]}")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)
plt.legend()

# Gráfico 2
plt.subplot(1, 2, 2)
plt.plot(eqm_plot2, label="Treinamento {plot2_data[0]}", color="orange")
plt.title(f"Erro Quadrático Médio (EQM) - Treinamento {plot2_data[0]}")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()""")
]

with open('PMC.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

# Create and save plot
eqm_plot1 = plot1_data[1]
eqm_plot2 = plot2_data[1]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(eqm_plot1, label=f"Treinamento {plot1_data[0]}", color="blue")
plt.title(f"Erro Quadrático Médio (EQM) - Treinamento {plot1_data[0]}")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eqm_plot2, label=f"Treinamento {plot2_data[0]}", color="orange")
plt.title(f"Erro Quadrático Médio (EQM) - Treinamento {plot2_data[0]}")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("grafico_eqm.png", dpi=150)
plt.close()

print("Process completed. README.md, PMC.ipynb and grafico_eqm.png have been created.")
