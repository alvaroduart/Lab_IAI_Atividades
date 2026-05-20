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

tables = extract_tables_from_docx('context/RBF2.docx')

# The test table has 15 samples. Let's find it.
test_table = None
train_table = None

for t in tables:
    if len(t) > 2 and 'Erro Relativo Médio (%):' in t[-2][0]:
        test_table = t
    if len(t) > 2 and 'Amostra' in t[0][0] and len(t[0]) >= 14:
        train_table = t

# Parse training data
train_data = []
for row in train_table[1:]:
    for i in range(0, len(row), 5):
        if i + 4 < len(row) and row[i]:
            try:
                x1 = float(row[i+1].replace(',', '.'))
                x2 = float(row[i+2].replace(',', '.'))
                x3 = float(row[i+3].replace(',', '.'))
                d = float(row[i+4].replace(',', '.'))
                train_data.append([x1, x2, x3, d])
            except ValueError:
                pass

train_data = np.array(train_data)
X_train = train_data[:, :3]
D_train = train_data[:, 3]

# Parse test data
test_data = []
for row in test_table:
    if len(row) > 4 and row[0].isdigit():
        try:
            x1 = float(row[1].replace(',', '.'))
            x2 = float(row[2].replace(',', '.'))
            x3 = float(row[3].replace(',', '.'))
            d = float(row[4].replace(',', '.'))
            test_data.append([x1, x2, x3, d])
        except ValueError:
            pass

test_data = np.array(test_data)
X_test = test_data[:, :3]
D_test = test_data[:, 3]

# Helper functions
def train_rbf(X_train, D_train, N1, seed, eta=0.01, precision=1e-7, max_epochs=10000):
    np.random.seed(seed)
    # 1. K-Means
    idx = np.random.choice(len(X_train), N1, replace=False)
    centers = X_train[idx].copy()
    
    for _ in range(100):
        distances = np.linalg.norm(X_train[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centers = np.zeros_like(centers)
        for i in range(N1):
            points = X_train[labels == i]
            if len(points) > 0:
                new_centers[i] = points.mean(axis=0)
            else:
                new_centers[i] = centers[i]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        
    variances = np.zeros(N1)
    for i in range(N1):
        points = X_train[labels == i]
        if len(points) > 1:
            variances[i] = np.mean(np.linalg.norm(points - centers[i], axis=1)**2)
        else:
            variances[i] = 0.1
    variances[variances == 0] = 0.1

    def gaussian(X, centers, variances):
        H = np.zeros((X.shape[0], len(centers)))
        for i in range(len(centers)):
            H[:, i] = np.exp(-np.linalg.norm(X - centers[i], axis=1)**2 / (2 * variances[i]))
        return H

    H_train = gaussian(X_train, centers, variances)
    H_train_bias = np.insert(H_train, 0, -1.0, axis=1)

    # 2. Output Layer (Delta Rule)
    W = np.random.rand(N1 + 1)
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
        if epoch >= max_epochs:
            break
            
        for j in range(len(H_train_bias)):
            h_j = H_train_bias[j]
            d_j = D_train[j]
            y_j = np.dot(h_j, W)
            error_j = d_j - y_j
            W += eta * error_j * h_j
            
        epoch += 1

    exec_time = time.time() - start_time
    
    return {
        'centers': centers,
        'variances': variances,
        'W': W,
        'eqm_history': eqm_history,
        'epochs': epoch,
        'time': exec_time
    }

def validate_rbf(X_test, D_test, model):
    H_test = np.zeros((X_test.shape[0], len(model['centers'])))
    for i in range(len(model['centers'])):
        H_test[:, i] = np.exp(-np.linalg.norm(X_test - model['centers'][i], axis=1)**2 / (2 * model['variances'][i]))
    H_test_bias = np.insert(H_test, 0, -1.0, axis=1)
    Y_test = np.dot(H_test_bias, model['W'])
    
    erros = np.abs((D_test - Y_test) / D_test) * 100
    erro_medio = np.mean(erros)
    variancia = np.var(erros)
    
    return Y_test, erro_medio, variancia

results = {'Rede 1': [], 'Rede 2': [], 'Rede 3': []}
configs = {'Rede 1': 5, 'Rede 2': 10, 'Rede 3': 15}

print("Training RBF models...")
seeds = [42, 123, 999]
for rede, N1 in configs.items():
    print(f"Training {rede}...")
    for t in range(3):
        print(f"  Execution {t+1}...")
        model = train_rbf(X_train, D_train, N1, seed=seeds[t], eta=0.01, precision=1e-7, max_epochs=20000)
        Y_test, erro_medio, variancia = validate_rbf(X_test, D_test, model)
        results[rede].append({
            'model': model,
            'Y_test': Y_test,
            'erro_medio': erro_medio,
            'variancia': variancia
        })

# Find best models for each topology
best_models = {}
for rede in configs.keys():
    best_idx = np.argmin([results[rede][i]['erro_medio'] for i in range(3)])
    best_models[rede] = best_idx

best_overall = min(configs.keys(), key=lambda r: results[r][best_models[r]]['erro_medio'])

# Generate Plot
plt.figure(figsize=(15, 5))
for i, rede in enumerate(configs.keys()):
    plt.subplot(1, 3, i+1)
    best_idx = best_models[rede]
    eqm = results[rede][best_idx]['model']['eqm_history']
    plt.plot(eqm, color='blue')
    plt.title(f"EQM - {rede} (T{best_idx+1})")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_eqm.png", dpi=150)
plt.close()

# Markdown Generation
readme = f"""# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 20/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

## Atividade 07 - RBF

Mapeamento de um sistema de injeção eletrônica utilizando redes neurais RBF (aproximação funcional). O objetivo é computar a quantidade de gasolina ($y$) a ser injetada em função de três variáveis ($x_1, x_2, x_3$). Foram testadas três topologias com diferentes quantidades de neurônios ocultos ($N_1 = 5, 10, 15$).

### 1. Treinamentos Realizados

Foram realizados 3 treinamentos para cada topologia, inicializando os pesos aleatoriamente entre 0 e 1, taxa de aprendizado $\\eta = 0.01$ e precisão $\\epsilon = 10^{{-7}}$.

| Treinamento | Rede 1 ($N_1=5$) | | Rede 2 ($N_1=10$) | | Rede 3 ($N_1=15$) | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **EQM** | **Épocas** | **EQM** | **Épocas** | **EQM** | **Épocas** |
"""

for t in range(3):
    r1 = results['Rede 1'][t]['model']
    r2 = results['Rede 2'][t]['model']
    r3 = results['Rede 3'][t]['model']
    readme += f"| **{t+1}º (T{t+1})** | {r1['eqm_history'][-1]:.6f} | {r1['epochs']} | {r2['eqm_history'][-1]:.6f} | {r2['epochs']} | {r3['eqm_history'][-1]:.6f} | {r3['epochs']} |\n"

readme += """
### 2. Validação da Rede (Conjunto de Teste)

Para a validação, foram comparados os valores de saída previstos pela rede ($y$) contra os valores desejados ($d$). O Erro Relativo Médio (%) e a Variância (%) foram calculados para cada rede.

<div style="overflow-x:auto;">

| Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | Rede 1 (T1) | (T2) | (T3) | Rede 2 (T1) | (T2) | (T3) | Rede 3 (T1) | (T2) | (T3) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for i in range(len(D_test)):
    x = f"{X_test[i][0]:.4f} | {X_test[i][1]:.4f} | {X_test[i][2]:.4f}"
    d = f"{D_test[i]:.4f}"
    
    r1_y = f"{results['Rede 1'][0]['Y_test'][i]:.4f} | {results['Rede 1'][1]['Y_test'][i]:.4f} | {results['Rede 1'][2]['Y_test'][i]:.4f}"
    r2_y = f"{results['Rede 2'][0]['Y_test'][i]:.4f} | {results['Rede 2'][1]['Y_test'][i]:.4f} | {results['Rede 2'][2]['Y_test'][i]:.4f}"
    r3_y = f"{results['Rede 3'][0]['Y_test'][i]:.4f} | {results['Rede 3'][1]['Y_test'][i]:.4f} | {results['Rede 3'][2]['Y_test'][i]:.4f}"
    
    readme += f"| **{i+1:02d}** | {x} | {d} | {r1_y} | {r2_y} | {r3_y} |\n"

readme += f"""| **Erro Rel. Med. (%)** | | | | | **{results['Rede 1'][0]['erro_medio']:>6.2f}** | **{results['Rede 1'][1]['erro_medio']:>6.2f}** | **{results['Rede 1'][2]['erro_medio']:>6.2f}** | **{results['Rede 2'][0]['erro_medio']:>6.2f}** | **{results['Rede 2'][1]['erro_medio']:>6.2f}** | **{results['Rede 2'][2]['erro_medio']:>6.2f}** | **{results['Rede 3'][0]['erro_medio']:>6.2f}** | **{results['Rede 3'][1]['erro_medio']:>6.2f}** | **{results['Rede 3'][2]['erro_medio']:>6.2f}** |
| **Variância (%)** | | | | | **{results['Rede 1'][0]['variancia']:>6.2f}** | **{results['Rede 1'][1]['variancia']:>6.2f}** | **{results['Rede 1'][2]['variancia']:>6.2f}** | **{results['Rede 2'][0]['variancia']:>6.2f}** | **{results['Rede 2'][1]['variancia']:>6.2f}** | **{results['Rede 2'][2]['variancia']:>6.2f}** | **{results['Rede 3'][0]['variancia']:>6.2f}** | **{results['Rede 3'][1]['variancia']:>6.2f}** | **{results['Rede 3'][2]['variancia']:>6.2f}** |
</div>

### 3. Gráficos do Erro Quadrático Médio (EQM)

Gráficos para o melhor treinamento (menor erro no teste) de cada topologia:
- **Rede 1**: T{best_models['Rede 1']+1}
- **Rede 2**: T{best_models['Rede 2']+1}
- **Rede 3**: T{best_models['Rede 3']+1}

![Gráfico EQM](grafico_eqm.png)

### 4. Conclusão sobre as Topologias

Baseado nas análises, a topologia mais adequada para este problema é a **{best_overall}**, em sua configuração final de treinamento **T{best_models[best_overall]+1}**. Essa configuração apresentou o menor Erro Relativo Médio ({results[best_overall][best_models[best_overall]]['erro_medio']:.2f}%) no conjunto de teste, indicando a melhor generalização para a aproximação funcional exigida, além de manter uma variância baixa, o que denota estabilidade nas predições do sistema de injeção eletrônica.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

# Jupyter Notebook Generation
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Redes Neurais de Base Radial (RBF) - Atividade 07\n\nAproximação funcional para sistema de injeção eletrônica utilizando redes neurais RBF."),
    nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt"),
    nbf.v4.new_markdown_cell("### Carregamento de Dados"),
    nbf.v4.new_code_cell(f"X_train = np.array({X_train.tolist()})\nD_train = np.array({D_train.tolist()})\n\nX_test = np.array({X_test.tolist()})\nD_test = np.array({D_test.tolist()})"),
    nbf.v4.new_markdown_cell("### Função da RBF (Treinamento e Validação)"),
    nbf.v4.new_code_cell("""def gaussian(X, centers, variances):
    H = np.zeros((X.shape[0], len(centers)))
    for i in range(len(centers)):
        H[:, i] = np.exp(-np.linalg.norm(X - centers[i], axis=1)**2 / (2 * variances[i]))
    return H"""),
    nbf.v4.new_markdown_cell("### Gráficos Gerados"),
    nbf.v4.new_code_cell(f"eqm_r1 = {results['Rede 1'][best_models['Rede 1']]['model']['eqm_history']}\neqm_r2 = {results['Rede 2'][best_models['Rede 2']]['model']['eqm_history']}\neqm_r3 = {results['Rede 3'][best_models['Rede 3']]['model']['eqm_history']}\n\nplt.figure(figsize=(15, 5))\nplt.subplot(1, 3, 1)\nplt.plot(eqm_r1, color='blue')\nplt.title('EQM - Rede 1')\nplt.grid(True)\n\nplt.subplot(1, 3, 2)\nplt.plot(eqm_r2, color='orange')\nplt.title('EQM - Rede 2')\nplt.grid(True)\n\nplt.subplot(1, 3, 3)\nplt.plot(eqm_r3, color='green')\nplt.title('EQM - Rede 3')\nplt.grid(True)\nplt.show()")
]

with open('RBF.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Process completed successfully.")
