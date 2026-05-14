import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import nbformat as nbf
import matplotlib.pyplot as plt
import time

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

tables = extract_tables_from_docx('context/PMC2.docx')

# Parse training data (Table 3)
train_data = []
for row in tables[3][1:]:
    # First side
    if len(row) >= 8 and row[0]:
        try:
            train_data.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
        except ValueError:
            pass
    # Second side
    if len(row) >= 16 and row[8]:
        try:
            train_data.append([float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15])])
        except ValueError:
            pass

train_data = np.array(train_data)
X_train = train_data[:, :4]
D_train = train_data[:, 4:7]
X_train_bias = np.insert(X_train, 0, -1.0, axis=1)

# Parse test data (Table 2)
test_data = []
for row in tables[2][1:19]:
    try:
        x1 = float(row[1])
        x2 = float(row[2])
        x3 = float(row[3])
        x4 = float(row[4])
        d1 = float(row[5])
        d2 = float(row[6])
        d3 = float(row[7])
        test_data.append([x1, x2, x3, x4, d1, d2, d3])
    except ValueError:
        pass

test_data = np.array(test_data)
X_test = test_data[:, :4]
D_test = test_data[:, 4:7]
X_test_bias = np.insert(X_test, 0, -1.0, axis=1)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

eta = 0.1
alpha = 0.9
precision = 1e-6
np.random.seed(42)

# Initial random weights
W1_init = np.random.rand(5, 15)
W2_init = np.random.rand(16, 3)

def train_network(use_momentum=False):
    W1 = W1_init.copy()
    W2 = W2_init.copy()
    
    prev_delta_W1 = np.zeros_like(W1)
    prev_delta_W2 = np.zeros_like(W2)
    
    eqm_history = []
    epoch = 0
    
    start_time = time.time()
    
    while True:
        # Calculate EQM
        H_total = sigmoid(np.dot(X_train_bias, W1))
        H_total_bias = np.insert(H_total, 0, -1.0, axis=1)
        Y_total = sigmoid(np.dot(H_total_bias, W2))
        
        error_total = D_train - Y_total
        eqm = np.mean(error_total ** 2)
        eqm_history.append(eqm)
        
        if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
            break
            
        # Online training
        for j in range(len(X_train_bias)):
            x_j = X_train_bias[j].reshape(1, 5)
            d_j = D_train[j].reshape(1, 3)
            
            # Forward
            u_h = np.dot(x_j, W1)
            h_j = sigmoid(u_h)
            h_j_bias = np.insert(h_j, 0, -1.0, axis=1) # 1x16
            u_y = np.dot(h_j_bias, W2)
            y_j = sigmoid(u_y) # 1x3
            
            # Backprop
            error_y = d_j - y_j # 1x3
            delta_out = error_y * sigmoid_derivative(y_j) # 1x3
            
            delta_h = sigmoid_derivative(h_j) * np.dot(delta_out, W2[1:].T) # 1x15
            
            delta_W2 = eta * np.dot(h_j_bias.T, delta_out)
            delta_W1 = eta * np.dot(x_j.T, delta_h)
            
            if use_momentum:
                delta_W2 += alpha * prev_delta_W2
                delta_W1 += alpha * prev_delta_W1
                prev_delta_W2 = delta_W2
                prev_delta_W1 = delta_W1
            
            W2 += delta_W2
            W1 += delta_W1
            
        epoch += 1
        
    exec_time = time.time() - start_time
    
    # Predict on test
    H_test = sigmoid(np.dot(X_test_bias, W1))
    H_test_bias = np.insert(H_test, 0, -1.0, axis=1)
    Y_test = sigmoid(np.dot(H_test_bias, W2))
    
    Y_test_post = np.round(Y_test)
    acertos = np.sum(np.all(Y_test_post == D_test, axis=1))
    taxa_acerto = acertos / len(D_test) * 100
    
    return {
        'eqm_history': eqm_history,
        'epochs': epoch,
        'time': exec_time,
        'Y_test_post': Y_test_post,
        'taxa_acerto': taxa_acerto,
        'W1': W1,
        'W2': W2
    }

print("Training Standard Backprop...")
res_std = train_network(use_momentum=False)

print("Training Backprop with Momentum...")
res_mom = train_network(use_momentum=True)

# Generate README.md
readme_content = f"""# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 14/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

## Atividade 04 - PMC

Para o processamento de bebidas, a equipe de engenheiros resolveu aplicar uma rede perceptron multicamadas como classificadora de padrões, visando identificar qual conservante (A, B ou C) será aplicado em determinado lote de bebida com base em 4 variáveis (teor de água, grau de acidez, temperatura e tensão superficial). A rede possui topologia 4-15-3.

### 1 e 2. Treinamentos da Rede Perceptron

Foram executados treinamentos utilizando Backpropagation Padrão e Backpropagation com Momentum.
As matrizes de pesos foram inicializadas com valores aleatórios entre 0 e 1, utilizou-se a função logística, taxa de aprendizado $\\eta = 0.1$, fator de momentum $\\alpha = 0.9$ e precisão $\\epsilon = 10^{{-6}}$.

| Algoritmo | Erro Quadrático Médio (EQM) | Número de Épocas | Tempo de Processamento (s) |
|:---|:---:|:---:|:---:|
| Backpropagation Padrão | {res_std['eqm_history'][-1]:.6f} | {res_std['epochs']} | {res_std['time']:.4f} |
| Backpropagation c/ Momentum | {res_mom['eqm_history'][-1]:.6f} | {res_mom['epochs']} | {res_mom['time']:.4f} |

### 3. Gráficos do Erro Quadrático Médio (EQM)

![Gráfico EQM](grafico_eqm.png)

Os gráficos também estão disponíveis no notebook `PMC.ipynb`.

### 4 e 5. Validação da Rede

O pós-processamento das saídas para valores discretos foi feito usando o arredondamento simétrico, de forma a classificar a amostra em um dos 3 tipos de conservante (Tipo A: 1 0 0, Tipo B: 0 1 0, Tipo C: 0 0 1).

| Amostra | x1 | x2 | x3 | x4 | Desejado (d1, d2, d3) | Padrão (y1, y2, y3) | Momentum (y1, y2, y3) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for i in range(len(D_test)):
    row_x = f"{X_test[i][0]:.4f} | {X_test[i][1]:.4f} | {X_test[i][2]:.4f} | {X_test[i][3]:.4f}"
    d_str = f"{int(D_test[i][0])}, {int(D_test[i][1])}, {int(D_test[i][2])}"
    y_std = f"{int(res_std['Y_test_post'][i][0])}, {int(res_std['Y_test_post'][i][1])}, {int(res_std['Y_test_post'][i][2])}"
    y_mom = f"{int(res_mom['Y_test_post'][i][0])}, {int(res_mom['Y_test_post'][i][1])}, {int(res_mom['Y_test_post'][i][2])}"
    readme_content += f"| **{i+1}** | {row_x} | {d_str} | {y_std} | {y_mom} |\n"

readme_content += f"""
**Taxa de Acerto - Backpropagation Padrão:** {res_std['taxa_acerto']:.2f}%  
**Taxa de Acerto - Backpropagation c/ Momentum:** {res_mom['taxa_acerto']:.2f}%
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Save Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(res_std['eqm_history'], color="blue")
plt.title("EQM - Backpropagation Padrão")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(res_mom['eqm_history'], color="orange")
plt.title("EQM - Backpropagation c/ Momentum")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)

plt.tight_layout()
plt.savefig("grafico_eqm.png", dpi=150)
plt.close()

# Create Jupyter Notebook
nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Perceptron Multicamadas (PMC) - Atividade 04\n\nClassificação de padrões para aplicação de conservantes utilizando Backpropagation com e sem Momentum."),
    nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt"),
    nbf.v4.new_markdown_cell("### Dados de Treinamento e Teste"),
    nbf.v4.new_code_cell(f"X_train = np.array({X_train.tolist()})\nD_train = np.array({D_train.tolist()})\n\nX_train_bias = np.insert(X_train, 0, -1.0, axis=1)\n\nX_test = np.array({X_test.tolist()})\nD_test = np.array({D_test.tolist()})\nX_test_bias = np.insert(X_test, 0, -1.0, axis=1)"),
    nbf.v4.new_markdown_cell("### Funções Auxiliares"),
    nbf.v4.new_code_cell("def sigmoid(x):\n    x = np.clip(x, -500, 500)\n    return 1 / (1 + np.exp(-x))\n\ndef sigmoid_derivative(y):\n    return y * (1 - y)"),
    nbf.v4.new_markdown_cell("### Gráficos gerados"),
    nbf.v4.new_code_cell(f"eqm_std = {res_std['eqm_history']}\neqm_mom = {res_mom['eqm_history']}\n\nplt.figure(figsize=(12, 5))\nplt.subplot(1, 2, 1)\nplt.plot(eqm_std, color='blue')\nplt.title('EQM - Backpropagation Padrão')\nplt.xlabel('Épocas')\nplt.ylabel('EQM')\nplt.grid(True)\n\nplt.subplot(1, 2, 2)\nplt.plot(eqm_mom, color='orange')\nplt.title('EQM - Backpropagation c/ Momentum')\nplt.xlabel('Épocas')\nplt.ylabel('EQM')\nplt.grid(True)\n\nplt.tight_layout()\nplt.show()")
]

with open('PMC.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Process completed successfully.")
