# Otimização de Portfólio de Crédito: Da Ilusão do Crescimento à Eficiência de Capital

### 🎯 A idea
A expansão agressiva de volume mascarou a deterioração da carteira; a rentabilidade real só é alcançada cortando a exposição aos grades F e G e restringindo produtos de longo prazo para perfis Prime.

### 💼 O Desafio de Negócio
A instituição focou em crescimento de carteira, penetrando segmentos de alto risco em busca de retornos nominais superiores. No entanto, métricas estáticas de inadimplência esconderam um problema estrutural: o prêmio de risco cobrado não estava cobrindo a severidade das perdas (LGD), gerando uma "ilusão de crescimento" onde ativos tóxicos inflavam o balanço sem gerar lucro econômico real.

### 🛠️ Metodologia Aplicada (Análise)
Para isolar a qualidade da originação e calcular o retorno real, evitei métricas de vaidade e apliquei técnicas de modelagem regulatória e valuation:
* **Análise de Safra (Vintage Analysis):** Para monitorar a maturação do risco isolando o efeito do crescimento do volume.
* **Cálculo de LGD (Loss Given Default):** Análise da distribuição bimodal de recuperações para estimar perdas severas.
* **Retorno Líquido Anualizado (NAR):** Cálculo do lucro real descontando a perda esperada e custos operacionais por sub-grade.
* **PSI (Population Stability Index):** Monitoramento de data drift para garantir a validade das regras atuais.

### 🔍 Principais Insights
1. **Destruição de Valor:** A partir do Grade C, o retorno ajustado ao risco torna-se negativo. Os lucros dos clientes 'A' subsidiam o prejuízo estrutural dos clientes 'F' e 'G'.
2. **Seleção Adversa em Prazos Longos:** Empréstimos de 60 meses apresentam o dobro da inadimplência acumulada dos de 36 meses no mesmo estágio de vida (MOB), indicando falha na precificação da duração.
3. **Irreversibilidade do Default:** A mediana da LGD é de 94,16%, indicando que a recuperação pós-default é estatisticamente improvável, exigindo rigor na entrada e não na cobrança.

### 🚀 Plano de Ação Recomendado
Com base na fronteira eficiente de risco mapeada, a estratégia propõe:
* **Suspensão Imediata:** Bloqueio de novas concessões para Grades F e G.
* **Revisão de Política:** Restringir o produto de 60 meses exclusivamente para clientes com Score A e B.
* **Foco Regional:** Ajuste da régua de aprovação para clusters geográficos com risco sistêmico identificado.

### 🛠️ Link para análise
[credit-risk-eda-v01.ipynb](/credit-risk-eda-v01.ipynb)

---
**Ferramentas:** Python, Pandas, Matplotlib, Seaborn, WoE/IV Framework.
*Esta análise foi desenhada seguindo as melhores práticas de governança de dados e frameworks regulatórios de risco.*
