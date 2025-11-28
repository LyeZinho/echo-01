# echo-01 — Simulações pedagógicas de espalhamento (ondas 2D)

Projeto educativo para explorar conceitos de espalhamento de ondas em 2D
usando um solucionador FDTD (diferenças finitas no tempo) para ondas escalares.
Este repositório foi criado para aprendizagem — não é um preditor RCS real nem
um guia para projeto ou otimização de furtividade militar.

IMPORTANTE: Não fornecemos instruções passo-a-passo nem detalhes técnicos de
aplicações sensíveis. O objetivo aqui é didático: demonstrar reflexão,
difração e padrões qualitativos de espalhamento.

## Estrutura do projeto

- `src/` — implementações da simulação FDTD e utilitários
- `examples/` — scripts de exemplo: `simulate_circle.py` gera animação e análises
- `requirements.txt` — dependências necessárias
- `README.md` — este arquivo

## Mapa de aprendizagem (recomendações)

1. Fundamentos: revisitar Maxwell (para contexto) e a versão escalar da
	 equação da onda (Helmholtz / wave equation) para entender propagação.
2. Condições de contorno: reflexão e transmissão em interfaces, impedância.
3. Métodos numéricos: diferenças finitas (FDTD) e métodos de contorno / MoM
	 em termos conceituais (quando cada um é adequado)
4. Malhas e resolução: relação entre comprimento de onda e passo de malha,
	 e como isso afeta precisão e custo computacional.
5. Validação: testar contra casos analíticos simples (onda plana, esfera/placa).

## Métodos numéricos (alto nível)

- FDTD (diferenças finitas no tempo): útil para simulações no domínio do tempo
	e estudo de transientes e pulsos.
- Métodos de elementos finitos (FEM): flexíveis para geometrias complexas e
	materiais heterogêneos (domínio da frequência ou do tempo).
- Method of Moments (MoM): bom para problemas de superfície e antenas
	(domínio da frequência).
- Optics / Ray Tracing / Physical Optics (PO): aproximações geométricas quando
	as dimensões são muito maiores que o comprimento de onda.

## Exemplo (simulação 2D)

O exemplo `examples/simulate_circle.py` demonstra:

- excitação por uma fonte pontual (pulso Gaussiano)
- obstáculo circular como região de impedância diferente
- camada de amortecimento simples nas bordas para reduzir reflexões
- captura de sinal refletido em posição de receptor e análise FFT

### Instalação

Recomenda-se criar um virtualenv e instalar dependências:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Rodando o exemplo

```powershell
python examples/simulate_circle.py
```

Saídas:
- `examples/outputs/wave_scatter_circle.gif` — animação do campo (qualitativa)
- `examples/outputs/receiver_signal.png` — sinal no receptor (tempo)
- `examples/outputs/receiver_spectrum.png` — FFT do sinal (qualitativa)

### GUI interativa

Existe também uma pequena GUI para explorar os efeitos de parâmetros em tempo
real. Rode o script abaixo para abrir a janela interativa (Tkinter):

```powershell
python examples/gui_simulator.py
```

Use os controles para iniciar/pausar a simulação, variar a freqüência da fonte
e o raio do obstáculo — a janela mostra o campo instantâneo e o sinal no
receptor em tempo real.

## Observações de interpretação

Este repositório mostra fenômenos qualitativos: reflexão especular, difração
em bordas e ressonâncias (dependendo da geometria e frequência). Não é um
substituto para análises rigorosas de Maxwell com materiais complexos.

## Referências (sugestões para estudo)

- D. J. Griffiths — Introduction to Electrodynamics (fundamentos)
- A. Taflove, S. Hagness — Computational Electrodynamics: The FDTD Method
- J. A. Kong — Electromagnetic Wave Theory
- Textos sobre métodos numéricos (FEM, MoM) e teoria do espalhamento