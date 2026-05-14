import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import nbformat as nbf
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

tables = extract_tables_from_docx('context/PMC3.docx')

train_series = [0]*100
for row in tables[3][1:]:
    for i in range(4):
        if len(row) > i*2 + 1 and row[i*2] and row[i*2+1]:
            t_str = row[i*2].replace('t =', '').strip()
            val_str = row[i*2+1].strip()
            if t_str.isdigit():
                t = int(t_str)
                train_series[t-1] = float(val_str)

test_series = [0]*20
for row in tables[2][2:22]:
    if len(row) >= 2 and row[0] and row[1]:
        t_str = row[0].replace('t =', '').strip()
        val_str = row[1].strip()
        if t_str.isdigit():
            t = int(t_str)
            test_series[t-101] = float(val_str)

S = np.array(train_series + test_series)

def create_dataset(series, p, start_t, end_t):
    X = []
    Y = []
    for t in range(start_t, end_t + 1):
        # t is 1-indexed. index in series is t-1
        idx = t - 1
        # inputs are x(t-1), x(t-2), ..., x(t-p)
        # indices: idx-1, idx-2, ..., idx-p
        x = [series[idx - k] for k in range(1, p + 1)]
        X.append(x)
        Y.append(series[idx])
    return np.array(X), np.array(Y)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

eta = 0.1
alpha = 0.8
precision = 0.5e-6

def train_tdnn(p, N1, seed_offset):
    np.random.seed(42 + seed_offset)
    
    # Dataset
    X_train, D_train = create_dataset(S, p, p + 1, 100)
    X_test, D_test = create_dataset(S, p, 101, 120)
    
    X_train_bias = np.insert(X_train, 0, -1.0, axis=1)
    X_test_bias = np.insert(X_test, 0, -1.0, axis=1)
    D_train = D_train.reshape(-1, 1)
    D_test = D_test.reshape(-1, 1)
    
    # Initialize weights
    W1 = np.random.rand(p + 1, N1)
    W2 = np.random.rand(N1 + 1, 1)
    
    prev_delta_W1 = np.zeros_like(W1)
    prev_delta_W2 = np.zeros_like(W2)
    
    eqm_history = []
    epoch = 0
    
    start_time = time.time()
    
    while True:
        # Forward pass train
        H_total = sigmoid(np.dot(X_train_bias, W1))
        H_total_bias = np.insert(H_total, 0, -1.0, axis=1)
        Y_total = sigmoid(np.dot(H_total_bias, W2))
        
        error_total = D_train - Y_total
        eqm = np.mean(error_total ** 2)
        eqm_history.append(eqm)
        
        if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= precision:
            break
            
        # Online training with momentum
        for j in range(len(X_train_bias)):
            x_j = X_train_bias[j].reshape(1, p + 1)
            d_j = D_train[j].reshape(1, 1)
            
            # Forward
            u_h = np.dot(x_j, W1)
            h_j = sigmoid(u_h)
            h_j_bias = np.insert(h_j, 0, -1.0, axis=1)
            u_y = np.dot(h_j_bias, W2)
            y_j = sigmoid(u_y)
            
            # Backprop
            error_y = d_j - y_j
            delta_out = error_y * sigmoid_derivative(y_j)
            
            delta_h = sigmoid_derivative(h_j) * np.dot(delta_out, W2[1:].T)
            
            delta_W2 = eta * np.dot(h_j_bias.T, delta_out) + alpha * prev_delta_W2
            delta_W1 = eta * np.dot(x_j.T, delta_h) + alpha * prev_delta_W1
            
            W2 += delta_W2
            W1 += delta_W1
            
            prev_delta_W2 = delta_W2
            prev_delta_W1 = delta_W1
            
        epoch += 1
        
    # Validation on test
    H_test = sigmoid(np.dot(X_test_bias, W1))
    H_test_bias = np.insert(H_test, 0, -1.0, axis=1)
    Y_test = sigmoid(np.dot(H_test_bias, W2))
    
    erros_relativos = np.abs(Y_test - D_test) / D_test * 100
    erro_relativo_medio = np.mean(erros_relativos)
    variancia = np.var(erros_relativos)
    
    return {
        'eqm': eqm_history[-1],
        'epochs': epoch,
        'Y_test': Y_test.flatten(),
        'D_test': D_test.flatten(),
        'erro_relativo_medio': erro_relativo_medio,
        'variancia': variancia,
        'eqm_history': eqm_history,
        'time': time.time() - start_time
    }

topologies = [
    {'name': 'Rede 1', 'p': 5, 'N1': 10},
    {'name': 'Rede 2', 'p': 10, 'N1': 15},
    {'name': 'Rede 3', 'p': 15, 'N1': 25}
]

results = {}
for i, topo in enumerate(topologies):
    results[topo['name']] = []
    print(f"Training {topo['name']}...")
    for j in range(3):
        res = train_tdnn(topo['p'], topo['N1'], seed_offset=(i*10 + j))
        results[topo['name']].append(res)
        print(f"  T{j+1}: EQM = {res['eqm']:.6f}, Epochs = {res['epochs']}")

# Find best training for each network (lowest relative error on test)
best_results = {}
for name in results:
    best_idx = np.argmin([r['erro_relativo_medio'] for r in results[name]])
    best_results[name] = {
        'idx': best_idx,
        'res': results[name][best_idx]
    }

# Plots
# Plot 1: EQM vs Epocas for the best trainings
plt.figure(figsize=(15, 4))
for i, topo in enumerate(topologies):
    name = topo['name']
    best_res = best_results[name]['res']
    best_t = best_results[name]['idx'] + 1
    
    plt.subplot(1, 3, i+1)
    plt.plot(best_res['eqm_history'], color=f'C{i}')
    plt.title(f"{name} (Melhor: T{best_t}) - EQM")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_eqm_melhores.png", dpi=150)
plt.close()

# Plot 2: Desired vs Estimated for the best trainings
plt.figure(figsize=(15, 4))
t_axis = np.arange(101, 121)
for i, topo in enumerate(topologies):
    name = topo['name']
    best_res = best_results[name]['res']
    best_t = best_results[name]['idx'] + 1
    
    plt.subplot(1, 3, i+1)
    plt.plot(t_axis, best_res['D_test'], marker='o', label='Desejado', color='black')
    plt.plot(t_axis, best_res['Y_test'], marker='x', linestyle='--', label='Estimado', color=f'C{i}')
    plt.title(f"{name} (Melhor: T{best_t}) - Previsões")
    plt.xlabel("Tempo (t)")
    plt.ylabel("Valor f(t)")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_previsoes.png", dpi=150)
plt.close()

# Generate README.md
readme_content = """# Centro Federal de Educação Tecnológica de Minas Gerais
**Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 14/05/2026  
**Aluno(a):** Álvaro Henrique Duarte Mendes  
**Valor:** 100% | **Nota:** _______

---

## Atividade 05 - PMC (TDNN)

O objetivo desta atividade é prever o comportamento futuro de uma série temporal do mercado financeiro utilizando uma rede neural perceptron multicamadas com topologia "Time Delay" (TDNN).

Foram testadas três topologias candidatas, variando o número de entradas $p$ (janela de tempo) e o número de neurônios na camada oculta $N1$:
- **Rede 1**: $p = 5$ entradas, $N1 = 10$ neurônios.
- **Rede 2**: $p = 10$ entradas, $N1 = 15$ neurônios.
- **Rede 3**: $p = 15$ entradas, $N1 = 25$ neurônios.

### 1 e 2. Treinamentos Realizados

Para cada rede, efetuou-se 3 treinamentos utilizando Backpropagation com Momentum. As matrizes de pesos foram inicializadas aleatoriamente, ativadas via função logística (sigmoid).
Parâmetros: $\\eta = 0.1$, $\\alpha = 0.8$, $\\epsilon = 0.5 \\times 10^{-6}$.

| Treinamento | Rede 1 (EQM) | Rede 1 (Épocas) | Rede 2 (EQM) | Rede 2 (Épocas) | Rede 3 (EQM) | Rede 3 (Épocas) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

for j in range(3):
    r1 = results['Rede 1'][j]
    r2 = results['Rede 2'][j]
    r3 = results['Rede 3'][j]
    readme_content += f"| **{j+1}º (T{j+1})** | {r1['eqm']:.6f} | {r1['epochs']} | {r2['eqm']:.6f} | {r2['epochs']} | {r3['eqm']:.6f} | {r3['epochs']} |\n"

readme_content += """
### 3. Validação da Rede (Conjunto de Teste: t=101 a 120)

| Amostra | Desejado f(t) | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

# Extract rows
D = best_results['Rede 1']['res']['D_test']
for i in range(20):
    row_str = f"| **t = {101+i}** | {D[i]:.4f} "
    for topo in topologies:
        for j in range(3):
            val = results[topo['name']][j]['Y_test'][i]
            row_str += f"| {val:.4f} "
    row_str += "|\n"
    readme_content += row_str

readme_content += "| **Erro Relativo Médio (%)** | | "
for topo in topologies:
    for j in range(3):
        readme_content += f"{results[topo['name']][j]['erro_relativo_medio']:.4f} | "
readme_content += "\n| **Variância (%)** | | "
for topo in topologies:
    for j in range(3):
        readme_content += f"{results[topo['name']][j]['variancia']:.4f} | "
readme_content += "\n\n"

# Determine best topology overall
best_overall = None
best_err = float('inf')
for name, best in best_results.items():
    if best['res']['erro_relativo_medio'] < best_err:
        best_err = best['res']['erro_relativo_medio']
        best_overall = name

best_t_overall = best_results[best_overall]['idx'] + 1

readme_content += f"""### 4. Gráficos do Erro Quadrático Médio (EQM) para o Melhor Treinamento

Considerando o menor Erro Relativo Médio de cada topologia, foram gerados os gráficos:

![Gráficos EQM](grafico_eqm_melhores.png)

### 5. Gráficos de Previsão (Desejado vs Estimado) para o Melhor Treinamento

![Gráficos de Previsão](grafico_previsoes.png)

### 6. Conclusão

Baseado nas análises das tabelas e gráficos, a topologia mais adequada para a realização de previsões neste processo é a **{best_overall}**, utilizando a configuração do **Treinamento T{best_t_overall}**. Esta configuração alcançou o menor Erro Relativo Médio no conjunto de teste, evidenciando uma melhor generalização da série temporal.

### 7. Comentários sobre as Variantes do Backpropagation

**a) Resilient-Propagation (RProp):**
É um algoritmo de treinamento em que a atualização dos pesos baseia-se apenas no *sinal* (direção) do gradiente de erro local, ignorando a sua magnitude. Isso resolve o problema de gradientes que se anulam em regiões planas (como nas extremidades da função sigmoid). Ao invés de uma taxa de aprendizado global, o RProp mantém um tamanho de passo adaptativo para cada peso individualmente. Se o sinal do gradiente se mantém, o passo aumenta; se ele se inverte, o passo diminui. A principal vantagem é uma convergência substancialmente mais rápida que o Backpropagation padrão, com menos necessidade de ajuste manual de parâmetros.

**b) Levenberg-Marquardt (LM):**
É uma técnica baseada em métodos de segunda ordem que aproxima o método de otimização de Newton. Para calcular as atualizações dos pesos, ela utiliza a matriz Jacobiana, calculando as derivadas primeiras do erro de rede para cada peso e viés. Sua grande vantagem é possuir uma taxa de convergência extremamente alta para redes de tamanho pequeno a médio, superando tanto o backpropagation padrão quanto o com momentum em velocidade de iteração. O custo associado é a alta exigência computacional e uso de memória para armazenar e inverter matrizes, tornando o método inviável para redes neurais de grande porte.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

# Generate Notebook
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# TDNN - Atividade 05\n\nPrevisão de série temporal financeira."),
    nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt"),
    nbf.v4.new_markdown_cell("Os resultados consolidados e os gráficos estão registrados no arquivo `README.md` e em `grafico_eqm_melhores.png` / `grafico_previsoes.png`.")
]
with open('PMC.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Process completed.")
