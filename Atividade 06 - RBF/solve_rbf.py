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

tables = extract_tables_from_docx('context/RBF1.docx')

# Parse training data (Table 5)
train_data = []
for row in tables[5][1:]:
    try:
        x1 = float(row[1].replace(',', '.'))
        x2 = float(row[2].replace(',', '.'))
        d = float(row[3].replace(',', '.'))
        train_data.append([x1, x2, d])
    except ValueError:
        pass

train_data = np.array(train_data)
X_train = train_data[:, :2]
D_train = train_data[:, 2]

# Parse test data (Table 4)
test_data = []
for row in tables[4][1:11]:
    try:
        x1 = float(row[1].replace(',', '.'))
        x2 = float(row[2].replace(',', '.'))
        d = float(row[3].replace(',', '.'))
        test_data.append([x1, x2, d])
    except ValueError:
        pass

test_data = np.array(test_data)
X_test = test_data[:, :2]
D_test = test_data[:, 2]

# 1. K-Means
np.random.seed(42)
X_train_rad = X_train[D_train == 1]
idx = np.random.choice(len(X_train_rad), 2, replace=False)
centers = X_train_rad[idx].copy()

while True:
    distances = np.linalg.norm(X_train_rad[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    
    new_centers = np.array([X_train_rad[labels == i].mean(axis=0) for i in range(2)])
    if np.allclose(centers, new_centers):
        break
    centers = new_centers

variances = np.array([np.mean(np.linalg.norm(X_train_rad[labels == i] - centers[i], axis=1)**2) for i in range(2)])

def gaussian(X, centers, variances):
    H = np.zeros((X.shape[0], len(centers)))
    for i in range(len(centers)):
        H[:, i] = np.exp(-np.linalg.norm(X - centers[i], axis=1)**2 / (2 * variances[i]))
    return H

H_train = gaussian(X_train, centers, variances)
H_train_bias = np.insert(H_train, 0, -1.0, axis=1)

# 2. Output layer training (Delta Rule)
W = np.random.rand(3)
eta = 0.01
precision = 1e-7

eqm_history = []
epoch = 0
start_time = time.time()

while True:
    Y = np.dot(H_train_bias, W)
    error = D_train - Y
    eqm = np.mean(error**2)
    eqm_history.append(eqm)
    
    if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
        break
        
    for j in range(len(H_train_bias)):
        h_j = H_train_bias[j]
        d_j = D_train[j]
        y_j = np.dot(h_j, W)
        error_j = d_j - y_j
        W += eta * error_j * h_j
        
    epoch += 1

exec_time = time.time() - start_time

# 3. Test phase
H_test = gaussian(X_test, centers, variances)
H_test_bias = np.insert(H_test, 0, -1.0, axis=1)
Y_test = np.dot(H_test_bias, W)
Y_test_post = np.sign(Y_test)
# Map 0 to 1 just in case
Y_test_post[Y_test_post == 0] = 1

acertos = np.sum(Y_test_post == D_test)
taxa_acerto = acertos / len(D_test) * 100

print(f"Centros: {centers}")
print(f"Variancias: {variances}")
print(f"Pesos: {W}")
print(f"Taxa de Acerto: {taxa_acerto}%")

# Generate README.md
readme_content = f"""# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 20/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

## Atividade 06 - RBF

A verificação da presença de radiação em determinados compostos nucleares é feita através da análise da concentração de duas variáveis ($x_1$ e $x_2$). A partir de 50 situações (40 para treinamento e 10 para teste), treinou-se uma rede neural RBF (Função de Base Radial) para classificação de padrões. A topologia utilizada possui 2 entradas, 2 neurônios na camada oculta e 1 saída.

### 1. Treinamento da Camada Escondida (k-means)

O treinamento da camada escondida (centros e variâncias dos clusters) foi realizado considerando apenas os padrões com presença de radiação (d = 1), utilizando o algoritmo k-means.

| Cluster | Centro ($x_1$, $x_2$) | Variância ($\\sigma^2$) |
|:---:|:---:|:---:|
| **1** | ({centers[0][0]:.4f}, {centers[0][1]:.4f}) | {variances[0]:.4f} |
| **2** | ({centers[1][0]:.4f}, {centers[1][1]:.4f}) | {variances[1]:.4f} |

### 2. Treinamento da Camada de Saída (Regra Delta)

A camada de saída foi treinada com a regra delta generalizada (LMS) com taxa de aprendizado $\\eta = 0.01$ e precisão $\\epsilon = 10^{{-7}}$.

| Peso | Valor |
|:---:|:---:|
| **W21,0 (Bias)** | {W[0]:.6f} |
| **W21,1** | {W[1]:.6f} |
| **W21,2** | {W[2]:.6f} |

* A rede convergiu em {epoch} épocas (EQM final: {eqm_history[-1]:.6f}, Tempo: {exec_time:.4f}s).

### 3. Validação da Rede (Conjunto de Teste)

O pós-processamento foi feito através da função sinal: $y_{{pós}} = \\text{{sgn}}(y)$.

| Amostra | $x_1$ | $x_2$ | $d$ | $y$ (Real) | $y_{{pós}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for i in range(len(D_test)):
    readme_content += f"| **{i+1}** | {X_test[i][0]:.4f} | {X_test[i][1]:.4f} | {int(D_test[i])} | {Y_test[i]:.4f} | {int(Y_test_post[i])} |\n"

readme_content += f"""
**Taxa de Acerto:** {taxa_acerto:.2f}%

### 4. Gráfico do Erro Quadrático Médio (EQM)

![Gráfico EQM](grafico_eqm.png)

### 5. Estratégias para Aumentar a Taxa de Acerto

Caso a rede não atinja uma acurácia desejada, as seguintes abordagens poderiam ser adotadas:
1. **Aumentar o Número de Clusters (Neurônios Ocultos):** Isso daria maior capacidade de representação das regiões de ativação para a classe de presença de radiação.
2. **K-Means com Todas as Classes:** Modificar o treinamento da camada oculta para que os centros representem melhor todo o espaço, e não apenas a classe de presença de radiação (embora o enunciado exija apenas a presença).
3. **Variâncias Individuais por Eixo:** Ao invés de uma única variância (isotrópica) para cada cluster, usar uma matriz de covariância para capturar relações direcionais dos dados.
4. **Tuning da Taxa de Aprendizado ($\\eta$):** Ajustar o valor ou aplicar um decaimento, o que pode levar a um ajuste mais fino dos pesos.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Save Plot
plt.figure(figsize=(8, 5))
plt.plot(eqm_history, color="blue")
plt.title("EQM - Treinamento da RBF (Camada de Saída)")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_eqm.png", dpi=150)
plt.close()

# Create Jupyter Notebook
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Redes Neurais de Base Radial (RBF) - Atividade 06\n\nClassificação de radiação utilizando clusterização por k-means e treinamento da camada de saída com regra delta."),
    nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt"),
    nbf.v4.new_markdown_cell("### Dados de Treinamento e Teste"),
    nbf.v4.new_code_cell(f"X_train = np.array({X_train.tolist()})\nD_train = np.array({D_train.tolist()})\n\nX_test = np.array({X_test.tolist()})\nD_test = np.array({D_test.tolist()})"),
    nbf.v4.new_markdown_cell("### Funções Auxiliares"),
    nbf.v4.new_code_cell("def gaussian(X, centers, variances):\n    H = np.zeros((X.shape[0], len(centers)))\n    for i in range(len(centers)):\n        H[:, i] = np.exp(-np.linalg.norm(X - centers[i], axis=1)**2 / (2 * variances[i]))\n    return H"),
    nbf.v4.new_markdown_cell("### Resultados da Camada Oculta"),
    nbf.v4.new_code_cell(f"centers = np.array({centers.tolist()})\nvariances = np.array({variances.tolist()})\nprint('Centros:\\n', centers)\nprint('Variâncias:\\n', variances)"),
    nbf.v4.new_markdown_cell("### Gráfico de Erro"),
    nbf.v4.new_code_cell(f"eqm_history = {eqm_history}\n\nplt.figure(figsize=(8, 5))\nplt.plot(eqm_history, color='blue')\nplt.title('EQM - Treinamento RBF')\nplt.xlabel('Épocas')\nplt.ylabel('EQM')\nplt.grid(True)\nplt.show()")
]

with open('RBF.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Process completed successfully.")
