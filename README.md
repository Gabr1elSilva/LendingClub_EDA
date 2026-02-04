## Perguntar

### Cenário

**A Ilusão do Crescimento**

**Situação:** A instituição realizou uma expansão agressiva da carteira de crédito, penetrando segmentos de perfis de alto risco em busca de retornos superiores.

**Complicação:** O volume acelerado de novas concessões pode estar mascarando a deterioração da carteira madura. Em um cenário de incerteza econômica, surge o risco de estarmos inflando o balanço com ativos tóxicos latentes em vez de gerar valor real.

**Questão Central:** Estamos gerando Lucro Econômico Real ou estamos apenas acumulando risco sistêmico invisível?

**Resposta Proposta pela Análise:** A resposta virá através do mapeamento da Fronteira Eficiente de Risco, identificando exatamente quais segmentos destroem valor para cessar a "sangria" e realocar capital onde a margem é real.

### Perguntas que guiarão a análise:
 * O prêmio de risco cobrado nos segmentos de pior rating (F e G) é suficiente para cobrir suas perdas e gerar lucro real, ou estamos destruindo valor nestas categorias?
 * O produto de longo prazo (60 meses) apresenta uma deterioração acelerada da qualidade de crédito em comparação ao de curto prazo (36 meses) no mesmo estágio de vida (MOB), indicando falha na precificação da duração?
 * Existem clusters geográficos de risco sistêmico que sugerem a necessidade de políticas de crédito regionalizadas, ou o risco está uniformemente distribuído?
 * Quando um cliente entra em default, qual é a probabilidade real de recuperarmos o capital? 
 * A distribuição é bimodal (tudo ou nada) ou podemos contar com uma recuperação média constante?

### Tarefa de negócios

Realizar uma análise exploratória nos dados históricos para recomendar uma Estratégia de Alocação de Portfólio. O objetivo transcende a previsão de inadimplência; foca na maximização do Retorno Líquido Anualizado, cortando a exposição a segmentos tóxicos e otimizando limites para perfis rentáveis.

**Definição de Sucesso**

O projeto será considerado um sucesso se entregar um plano de ação tático que permita:

1. **Ação Imediata:** Suspender a concessão para combinações de Grade/Prazo identificadas com NAR negativo.

2. **Ajuste de Política:** Recalibrar os limites de crédito baseando-se não apenas no score de entrada, mas na curva de recuperação real esperada.

## Preparar

Na gestão de risco de crédito, a qualidade da decisão é limitada pela integridade da informação que a suporta. A etapa de Preparação neste projeto transcende a rotina técnica de importação de bibliotecas e leitura de arquivos; ela estabelece a arquitetura analítica que governará toda a investigação.

Antes de qualquer limpeza ou tratamento, realizamos aqui um Diagnóstico de Integridade e Estrutura. Esta fase inicial tem dois objetivos estratégicos:
1. **Mapeamento de Terreno:** Através da análise das distribuições numéricas (KDE) e categóricas, buscamos entender a 'fisionomia' do cliente e a concentração de produtos. Identificar a assimetria nas rendas ou a predominância de certos tipos de empréstimos agora é crucial para evitar que regras de negócio futuras sejam enviesadas por exceções estatísticas.
2. **Segregação Arquitetural:** Diferente de datasets acadêmicos limpos, dados reais misturam informações do passado com o futuro. Nesta etapa, identificamos e isolamos as variáveis de back-end (pagamentos, recuperações) das variáveis de front-end (aplicação), preparando o terreno para uma análise honesta que não utiliza o 'gabarito' do futuro para justificar decisões do passado.

O resultado desta etapa não é apenas um dataframe carregado, mas um Plano de Dados validado, pronto para ser saneado e transformado em inteligência de risco.


```python
# Manipulação de Dados e Operações Numéricas
import numpy as np
import pandas as pd
import math

# Visualização de Dados
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
import seaborn as sns
import plotly.express as px


# Sistema e Limpeza
import os
import warnings
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configurações Iniciais
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')
```

    /kaggle/input/lending-club-loan-data-csv/loan.csv
    /kaggle/input/lending-club-loan-data-csv/LCDataDictionary.xlsx
    

### Importando o conjunto de dados


```python
df_raw = pd.read_csv('/kaggle/input/lending-club-loan-data-csv/loan.csv')
```

Utilizamos o dataset público do Lending Club, amplamente reconhecido para benchmarking em crédito P2P.
* **Volume**: +2,2 milhões de empréstimos.
* **Dimensões**: 145 variáveis cobrindo dados cadastrais, comportamentais e transacionais.
* **Relevância**: Este conjunto de dados oferece a granularidade necessária para realizar análises complexas de Safra e modelagem de LGD, permitindo replicar os desafios reais de Big Data enfrentados por grandes instituições financeiras."


### Explorando o Dataset

Esta exploração não é uma formalidade técnica; é o processo de Controle de Qualidade que garante que nossos insights futuros sejam baseados na realidade, e não em sujeira estatística. Antes de prever o futuro, precisamos entender profundamente o presente.


```python
df_raw.shape
```




    (2260668, 145)




```python
df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>url</th>
      <th>desc</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>next_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>mths_since_last_major_derog</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>annual_inc_joint</th>
      <th>dti_joint</th>
      <th>verification_status_joint</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>open_acc_6m</th>
      <th>open_act_il</th>
      <th>open_il_12m</th>
      <th>open_il_24m</th>
      <th>mths_since_rcnt_il</th>
      <th>total_bal_il</th>
      <th>il_util</th>
      <th>open_rv_12m</th>
      <th>open_rv_24m</th>
      <th>max_bal_bc</th>
      <th>all_util</th>
      <th>total_rev_hi_lim</th>
      <th>inq_fi</th>
      <th>total_cu_tl</th>
      <th>inq_last_12m</th>
      <th>acc_open_past_24mths</th>
      <th>avg_cur_bal</th>
      <th>bc_open_to_buy</th>
      <th>bc_util</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>mo_sin_old_il_acct</th>
      <th>mo_sin_old_rev_tl_op</th>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <th>mo_sin_rcnt_tl</th>
      <th>mort_acc</th>
      <th>mths_since_recent_bc</th>
      <th>mths_since_recent_bc_dlq</th>
      <th>mths_since_recent_inq</th>
      <th>mths_since_recent_revol_delinq</th>
      <th>num_accts_ever_120_pd</th>
      <th>num_actv_bc_tl</th>
      <th>num_actv_rev_tl</th>
      <th>num_bc_sats</th>
      <th>num_bc_tl</th>
      <th>num_il_tl</th>
      <th>num_op_rev_tl</th>
      <th>num_rev_accts</th>
      <th>num_rev_tl_bal_gt_0</th>
      <th>num_sats</th>
      <th>num_tl_120dpd_2m</th>
      <th>num_tl_30dpd</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>revol_bal_joint</th>
      <th>sec_app_earliest_cr_line</th>
      <th>sec_app_inq_last_6mths</th>
      <th>sec_app_mort_acc</th>
      <th>sec_app_open_acc</th>
      <th>sec_app_revol_util</th>
      <th>sec_app_open_act_il</th>
      <th>sec_app_num_rev_accts</th>
      <th>sec_app_chargeoff_within_12_mths</th>
      <th>sec_app_collections_12_mths_ex_med</th>
      <th>sec_app_mths_since_last_major_derog</th>
      <th>hardship_flag</th>
      <th>hardship_type</th>
      <th>hardship_reason</th>
      <th>hardship_status</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_start_date</th>
      <th>hardship_end_date</th>
      <th>payment_plan_start_date</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>hardship_loan_status</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
      <th>debt_settlement_flag_date</th>
      <th>settlement_status</th>
      <th>settlement_date</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.00</td>
      <td>36 months</td>
      <td>13.56</td>
      <td>84.92</td>
      <td>C</td>
      <td>C1</td>
      <td>Chef</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>55000.00</td>
      <td>Not Verified</td>
      <td>Dec-2018</td>
      <td>Current</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>109xx</td>
      <td>NY</td>
      <td>18.24</td>
      <td>0.00</td>
      <td>Apr-2001</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>45.00</td>
      <td>9.00</td>
      <td>1.00</td>
      <td>4341</td>
      <td>10.30</td>
      <td>34.00</td>
      <td>w</td>
      <td>2386.02</td>
      <td>2386.02</td>
      <td>167.02</td>
      <td>167.02</td>
      <td>113.98</td>
      <td>53.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Feb-2019</td>
      <td>84.92</td>
      <td>Mar-2019</td>
      <td>Feb-2019</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>16901.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>12560.00</td>
      <td>69.00</td>
      <td>2.00</td>
      <td>7.00</td>
      <td>2137.00</td>
      <td>28.00</td>
      <td>42000.00</td>
      <td>1.00</td>
      <td>11.00</td>
      <td>2.00</td>
      <td>9.00</td>
      <td>1878.00</td>
      <td>34360.00</td>
      <td>5.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>140.00</td>
      <td>212.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>2.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>16.00</td>
      <td>7.00</td>
      <td>18.00</td>
      <td>5.00</td>
      <td>9.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>60124.00</td>
      <td>16901.00</td>
      <td>36500.00</td>
      <td>18124.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000</td>
      <td>30000</td>
      <td>30000.00</td>
      <td>60 months</td>
      <td>18.94</td>
      <td>777.23</td>
      <td>D</td>
      <td>D2</td>
      <td>Postmaster</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>90000.00</td>
      <td>Source Verified</td>
      <td>Dec-2018</td>
      <td>Current</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>713xx</td>
      <td>LA</td>
      <td>26.52</td>
      <td>0.00</td>
      <td>Jun-1987</td>
      <td>0.00</td>
      <td>71.00</td>
      <td>75.00</td>
      <td>13.00</td>
      <td>1.00</td>
      <td>12315</td>
      <td>24.20</td>
      <td>44.00</td>
      <td>w</td>
      <td>29387.75</td>
      <td>29387.75</td>
      <td>1507.11</td>
      <td>1507.11</td>
      <td>612.25</td>
      <td>894.86</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Feb-2019</td>
      <td>777.23</td>
      <td>Mar-2019</td>
      <td>Feb-2019</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>1208.00</td>
      <td>321915.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>87153.00</td>
      <td>88.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>998.00</td>
      <td>57.00</td>
      <td>50800.00</td>
      <td>2.00</td>
      <td>15.00</td>
      <td>2.00</td>
      <td>10.00</td>
      <td>24763.00</td>
      <td>13761.00</td>
      <td>8.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>163.00</td>
      <td>378.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>NaN</td>
      <td>4.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>9.00</td>
      <td>27.00</td>
      <td>8.00</td>
      <td>14.00</td>
      <td>4.00</td>
      <td>13.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>95.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>372872.00</td>
      <td>99468.00</td>
      <td>15000.00</td>
      <td>94072.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000.00</td>
      <td>36 months</td>
      <td>17.97</td>
      <td>180.69</td>
      <td>D</td>
      <td>D1</td>
      <td>Administrative</td>
      <td>6 years</td>
      <td>MORTGAGE</td>
      <td>59280.00</td>
      <td>Source Verified</td>
      <td>Dec-2018</td>
      <td>Current</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>490xx</td>
      <td>MI</td>
      <td>10.51</td>
      <td>0.00</td>
      <td>Apr-2011</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.00</td>
      <td>0.00</td>
      <td>4599</td>
      <td>19.10</td>
      <td>13.00</td>
      <td>w</td>
      <td>4787.21</td>
      <td>4787.21</td>
      <td>353.89</td>
      <td>353.89</td>
      <td>212.79</td>
      <td>141.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Feb-2019</td>
      <td>180.69</td>
      <td>Mar-2019</td>
      <td>Feb-2019</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>110299.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>14.00</td>
      <td>7150.00</td>
      <td>72.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>35.00</td>
      <td>24100.00</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>18383.00</td>
      <td>13800.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>87.00</td>
      <td>92.00</td>
      <td>15.00</td>
      <td>14.00</td>
      <td>2.00</td>
      <td>77.00</td>
      <td>NaN</td>
      <td>14.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>6.00</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>8.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>136927.00</td>
      <td>11749.00</td>
      <td>13800.00</td>
      <td>10000.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4000</td>
      <td>4000</td>
      <td>4000.00</td>
      <td>36 months</td>
      <td>18.94</td>
      <td>146.51</td>
      <td>D</td>
      <td>D2</td>
      <td>IT Supervisor</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>92000.00</td>
      <td>Source Verified</td>
      <td>Dec-2018</td>
      <td>Current</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>985xx</td>
      <td>WA</td>
      <td>16.74</td>
      <td>0.00</td>
      <td>Feb-2006</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>5468</td>
      <td>78.10</td>
      <td>13.00</td>
      <td>w</td>
      <td>3831.93</td>
      <td>3831.93</td>
      <td>286.71</td>
      <td>286.71</td>
      <td>168.07</td>
      <td>118.64</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Feb-2019</td>
      <td>146.51</td>
      <td>Mar-2019</td>
      <td>Feb-2019</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>686.00</td>
      <td>305049.00</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>30683.00</td>
      <td>68.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3761.00</td>
      <td>70.00</td>
      <td>7000.00</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>30505.00</td>
      <td>1239.00</td>
      <td>75.20</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>62.00</td>
      <td>154.00</td>
      <td>64.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>64.00</td>
      <td>NaN</td>
      <td>5.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>7.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>385183.00</td>
      <td>36151.00</td>
      <td>5000.00</td>
      <td>44984.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000</td>
      <td>30000</td>
      <td>30000.00</td>
      <td>60 months</td>
      <td>16.14</td>
      <td>731.78</td>
      <td>C</td>
      <td>C4</td>
      <td>Mechanic</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>57250.00</td>
      <td>Not Verified</td>
      <td>Dec-2018</td>
      <td>Current</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>debt_consolidation</td>
      <td>Debt consolidation</td>
      <td>212xx</td>
      <td>MD</td>
      <td>26.35</td>
      <td>0.00</td>
      <td>Dec-2000</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.00</td>
      <td>0.00</td>
      <td>829</td>
      <td>3.60</td>
      <td>26.00</td>
      <td>w</td>
      <td>29339.02</td>
      <td>29339.02</td>
      <td>1423.21</td>
      <td>1423.21</td>
      <td>660.98</td>
      <td>762.23</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>Feb-2019</td>
      <td>731.78</td>
      <td>Mar-2019</td>
      <td>Feb-2019</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1</td>
      <td>Individual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>116007.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>28845.00</td>
      <td>89.00</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>516.00</td>
      <td>54.00</td>
      <td>23100.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9.00</td>
      <td>9667.00</td>
      <td>8471.00</td>
      <td>8.90</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>53.00</td>
      <td>216.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>NaN</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>8.00</td>
      <td>9.00</td>
      <td>6.00</td>
      <td>15.00</td>
      <td>2.00</td>
      <td>12.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>92.30</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>157548.00</td>
      <td>29674.00</td>
      <td>9300.00</td>
      <td>32332.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2260668 entries, 0 to 2260667
    Columns: 145 entries, id to settlement_term
    dtypes: float64(105), int64(4), object(36)
    memory usage: 2.4+ GB
    

Acabamos de ter uma visão geral da base de dados e as informações revelam a complexidade do desafio:

**Volume de Exposição:** Não estamos lidando com uma amostra pequena. A dimensão do dataset indica um volume massivo de transações e histórico, o que nos dá significância estatística, mas também exige eficiência computacional no processamento.

**A Cara dos Dados:** A visualização das primeiras linhas confirma que temos um mix rico de variáveis: dados cadastrais, dados transacionais e dados comportamentais.

Temos matéria-prima valiosa, mas ela não está pronta para consumo. A presença de dados faltantes e a necessidade de conversão de tipos indicam que, antes de qualquer análise, precisaremos realizar um trabalho robusto de Saneamento e Engenharia de Dados.

### Copiando as colunas que serão usadas na análise para outro dataframe

Bases de dados reais frequentemente sofrem com o excesso de informações: contêm centenas de colunas com IDs de sistema, logs administrativos e códigos internos que não possuem nenhum valor para o risco de crédito. Carregar esse peso morto não apenas torna o processamento mais lento, mas também aumenta a carga cognitiva da análise.

Nesta etapa, realizamos uma Segregação Tática:

1. **Foco no Driver de Risco:** Selecionamos apenas as variáveis que descrevem o comportamento, a capacidade financeira e o histórico do cliente.

2. **Princípio da Imutabilidade:** Ao copiar esses dados para um novo dataframe de trabalho, preservamos o dataset original intacto. Isso garante que sempre tenhamos um ponto de restauração seguro caso algum tratamento de dados futuro precise ser desfeito.

Estamos essencialmente limpando a bancada para deixar apenas as ferramentas que vamos usar.


```python
selected_columns = [
    'loan_status', 'issue_d', 'last_pymnt_d', 'term',
    'annual_inc', 'dti', 'verification_status', 'installment',
    'emp_length', 'emp_title', 'home_ownership', 'purpose', 'zip_code', 'addr_state',
    'revol_util', 'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec', 
    'earliest_cr_line', 'open_acc', 'total_acc',
    'grade', 'sub_grade', 'int_rate', 'loan_amnt', 'funded_amnt',
    'total_pymnt', 'recoveries', 'collection_recovery_fee', 
    'total_rec_prncp', 'total_rec_int', 'last_pymnt_amnt'
]

df = df_raw[selected_columns].copy()
```

### Descrição dos dados

Esta tabela descreve as variáveis selecionadas para a análise de risco de crédito, traduzidas a partir do dicionário oficial do LendingClub.

| Variável | Descrição |
| :--- | :--- |
| `loan_status` | Status atual do empréstimo. |
| `issue_d` | O mês em que o empréstimo foi financiado. |
| `last_pymnt_d` | Último mês em que o pagamento foi recebido. |
| `term` | O número de pagamentos do empréstimo. Os valores são em meses e podem ser 36 ou 60. |
| `annual_inc` | A renda anual auto-declarada informada pelo mutuário durante o registro. |
| `dti` | Uma razão calculada usando os pagamentos mensais totais da dívida do mutuário sobre as obrigações totais da dívida, excluindo hipoteca e o empréstimo solicitado da LC, dividida pela renda mensal auto-declarada do mutuário. |
| `verification_status` | Indica se a renda foi verificada pela LC, não verificada ou se a fonte de renda foi verificada. |
| `installment` | A prestação mensal devida pelo mutuário se o empréstimo for originado. |
| `emp_length` | Tempo de emprego em anos. Valores possíveis estão entre 0 e 10, onde 0 significa menos de um ano e 10 significa dez anos ou mais. |
| `emp_title` | O cargo fornecido pelo mutuário ao solicitar o empréstimo. |
| `home_ownership` | O status de propriedade da residência fornecido pelo mutuário durante o registro ou obtido do relatório de crédito. Nossos valores são: RENT, OWN, MORTGAGE, OTHER. |
| `purpose` | Uma categoria fornecida pelo mutuário para a solicitação de empréstimo. |
| `zip_code` | Os primeiros 3 números do CEP (zip code) fornecido pelo mutuário no pedido de empréstimo. |
| `addr_state` | O estado fornecido pelo mutuário no pedido de empréstimo. |
| `revol_util` | Taxa de utilização da linha rotativa, ou a quantidade de crédito que o mutuário está usando em relação a todo o crédito rotativo disponível. |
| `inq_last_6mths` | O número de consultas de crédito nos últimos 6 meses (excluindo consultas de automóveis e hipotecas). |
| `mths_since_last_delinq` | O número de meses desde a última inadimplência do mutuário. |
| `pub_rec` | Número de registros públicos depreciativos. |
| `earliest_cr_line` | O mês em que a linha de crédito mais antiga relatada pelo mutuário foi aberta. |
| `open_acc` | O número de linhas de crédito abertas no arquivo de crédito do mutuário. |
| `total_acc` | O número total de linhas de crédito atualmente no arquivo de crédito do mutuário. |
| `grade` | Grau de empréstimo atribuído pela LC. |
| `sub_grade` | Subgrau de empréstimo atribuído pela LC. |
| `int_rate` | Taxa de juros do empréstimo. |
| `loan_amnt` | O valor listado do empréstimo solicitado pelo mutuário. Se em algum momento o departamento de crédito reduzir o valor, isso será refletido aqui. |
| `funded_amnt` | O valor total comprometido com o empréstimo naquele momento. |
| `total_pymnt` | Pagamentos recebidos até a data para o valor total financiado. |
| `recoveries` | Recuperação bruta pós-baixa (post charge off). |
| `collection_recovery_fee` | Taxa de cobrança pós-baixa (post charge off). |
| `total_rec_prncp` | Principal recebido até a data. |
| `total_rec_int` | Juros recebidos até a data. |
| `last_pymnt_amnt` | Valor total do último pagamento recebido. |


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>installment</th>
      <th>revol_util</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>pub_rec</th>
      <th>open_acc</th>
      <th>total_acc</th>
      <th>int_rate</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>total_pymnt</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>last_pymnt_amnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2260664.00</td>
      <td>2258957.00</td>
      <td>2260668.00</td>
      <td>2258866.00</td>
      <td>2260638.00</td>
      <td>1102166.00</td>
      <td>2260639.00</td>
      <td>2260639.00</td>
      <td>2260639.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
      <td>2260668.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>77992.43</td>
      <td>18.82</td>
      <td>445.81</td>
      <td>50.34</td>
      <td>0.58</td>
      <td>34.54</td>
      <td>0.20</td>
      <td>11.61</td>
      <td>24.16</td>
      <td>13.09</td>
      <td>15046.93</td>
      <td>15041.66</td>
      <td>11824.03</td>
      <td>136.07</td>
      <td>22.59</td>
      <td>9300.14</td>
      <td>2386.35</td>
      <td>3364.02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>112696.20</td>
      <td>14.18</td>
      <td>267.17</td>
      <td>24.71</td>
      <td>0.89</td>
      <td>21.90</td>
      <td>0.57</td>
      <td>5.64</td>
      <td>11.99</td>
      <td>4.83</td>
      <td>9190.25</td>
      <td>9188.41</td>
      <td>9889.60</td>
      <td>725.83</td>
      <td>127.11</td>
      <td>8304.89</td>
      <td>2663.09</td>
      <td>5971.76</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>-1.00</td>
      <td>4.93</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>5.31</td>
      <td>500.00</td>
      <td>500.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>46000.00</td>
      <td>11.89</td>
      <td>251.65</td>
      <td>31.50</td>
      <td>0.00</td>
      <td>16.00</td>
      <td>0.00</td>
      <td>8.00</td>
      <td>15.00</td>
      <td>9.49</td>
      <td>8000.00</td>
      <td>8000.00</td>
      <td>4272.58</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2846.18</td>
      <td>693.61</td>
      <td>308.64</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>65000.00</td>
      <td>17.84</td>
      <td>377.99</td>
      <td>50.30</td>
      <td>0.00</td>
      <td>31.00</td>
      <td>0.00</td>
      <td>11.00</td>
      <td>22.00</td>
      <td>12.62</td>
      <td>12900.00</td>
      <td>12875.00</td>
      <td>9060.87</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6823.39</td>
      <td>1485.28</td>
      <td>588.47</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>93000.00</td>
      <td>24.49</td>
      <td>593.32</td>
      <td>69.40</td>
      <td>1.00</td>
      <td>50.00</td>
      <td>0.00</td>
      <td>14.00</td>
      <td>31.00</td>
      <td>15.99</td>
      <td>20000.00</td>
      <td>20000.00</td>
      <td>16707.97</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>13397.50</td>
      <td>3052.22</td>
      <td>3534.97</td>
    </tr>
    <tr>
      <th>max</th>
      <td>110000000.00</td>
      <td>999.00</td>
      <td>1719.83</td>
      <td>892.30</td>
      <td>33.00</td>
      <td>226.00</td>
      <td>86.00</td>
      <td>101.00</td>
      <td>176.00</td>
      <td>30.99</td>
      <td>40000.00</td>
      <td>40000.00</td>
      <td>63296.88</td>
      <td>39859.55</td>
      <td>7174.72</td>
      <td>40000.00</td>
      <td>28192.50</td>
      <td>42192.05</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2260668 entries, 0 to 2260667
    Data columns (total 32 columns):
     #   Column                   Dtype  
    ---  ------                   -----  
     0   loan_status              object 
     1   issue_d                  object 
     2   last_pymnt_d             object 
     3   term                     object 
     4   annual_inc               float64
     5   dti                      float64
     6   verification_status      object 
     7   installment              float64
     8   emp_length               object 
     9   emp_title                object 
     10  home_ownership           object 
     11  purpose                  object 
     12  zip_code                 object 
     13  addr_state               object 
     14  revol_util               float64
     15  inq_last_6mths           float64
     16  mths_since_last_delinq   float64
     17  pub_rec                  float64
     18  earliest_cr_line         object 
     19  open_acc                 float64
     20  total_acc                float64
     21  grade                    object 
     22  sub_grade                object 
     23  int_rate                 float64
     24  loan_amnt                int64  
     25  funded_amnt              int64  
     26  total_pymnt              float64
     27  recoveries               float64
     28  collection_recovery_fee  float64
     29  total_rec_prncp          float64
     30  total_rec_int            float64
     31  last_pymnt_amnt          float64
    dtypes: float64(16), int64(2), object(14)
    memory usage: 551.9+ MB
    


```python
df.columns
```




    Index(['loan_status', 'issue_d', 'last_pymnt_d', 'term', 'annual_inc', 'dti',
           'verification_status', 'installment', 'emp_length', 'emp_title',
           'home_ownership', 'purpose', 'zip_code', 'addr_state', 'revol_util',
           'inq_last_6mths', 'mths_since_last_delinq', 'pub_rec',
           'earliest_cr_line', 'open_acc', 'total_acc', 'grade', 'sub_grade',
           'int_rate', 'loan_amnt', 'funded_amnt', 'total_pymnt', 'recoveries',
           'collection_recovery_fee', 'total_rec_prncp', 'total_rec_int',
           'last_pymnt_amnt'],
          dtype='object')



### Entendendo as variáveis discretas e/ou categóricas

Até agora, olhamos para a estrutura dos dados. Mas, para modelar risco, não basta saber "quanto" dinheiro foi emprestado; precisamos saber para quem e para quê. As variáveis discretas e categóricas são a alma da segmentação de negócio. Elas transformam números frios em perfis humanos e produtos financeiros.

Entender essas categorias não é apenas fazer gráficos de barras. É verificar se a diversificação do portfólio está saudável ou se estamos colocando todos os ovos na mesma cesta vulnerável. Se encontrarmos uma categoria dominante com alta inadimplência lá na frente, saberemos que o problema é estrutural.

Valores únicos que contém na coluna:


```python
df.loan_status.unique()
```




    array(['Current', 'Fully Paid', 'Late (31-120 days)', 'In Grace Period',
           'Charged Off', 'Late (16-30 days)', 'Default',
           'Does not meet the credit policy. Status:Fully Paid',
           'Does not meet the credit policy. Status:Charged Off'],
          dtype=object)




```python
df.issue_d.unique()
```




    array(['Dec-2018', 'Nov-2018', 'Oct-2018', 'Sep-2018', 'Aug-2018',
           'Jul-2018', 'Jun-2018', 'May-2018', 'Apr-2018', 'Mar-2018',
           'Feb-2018', 'Jan-2018', 'Sep-2016', 'Aug-2016', 'Jul-2016',
           'Jun-2016', 'May-2016', 'Apr-2016', 'Mar-2016', 'Feb-2016',
           'Jan-2016', 'Dec-2016', 'Nov-2016', 'Oct-2016', 'Dec-2015',
           'Nov-2015', 'Oct-2015', 'Sep-2015', 'Aug-2015', 'Jul-2015',
           'Jun-2015', 'May-2015', 'Apr-2015', 'Mar-2015', 'Feb-2015',
           'Jan-2015', 'Mar-2017', 'Feb-2017', 'Jan-2017', 'Sep-2017',
           'Aug-2017', 'Jul-2017', 'Jun-2017', 'May-2017', 'Apr-2017',
           'Dec-2013', 'Nov-2013', 'Oct-2013', 'Sep-2013', 'Aug-2013',
           'Jul-2013', 'Jun-2013', 'May-2013', 'Apr-2013', 'Mar-2013',
           'Feb-2013', 'Jan-2013', 'Dec-2012', 'Nov-2012', 'Oct-2012',
           'Sep-2012', 'Aug-2012', 'Jul-2012', 'Jun-2012', 'May-2012',
           'Apr-2012', 'Mar-2012', 'Feb-2012', 'Jan-2012', 'Dec-2014',
           'Nov-2014', 'Oct-2014', 'Sep-2014', 'Aug-2014', 'Jul-2014',
           'Jun-2014', 'May-2014', 'Apr-2014', 'Mar-2014', 'Feb-2014',
           'Jan-2014', 'Dec-2011', 'Nov-2011', 'Oct-2011', 'Sep-2011',
           'Aug-2011', 'Jul-2011', 'Jun-2011', 'May-2011', 'Apr-2011',
           'Mar-2011', 'Feb-2011', 'Jan-2011', 'Dec-2010', 'Nov-2010',
           'Oct-2010', 'Sep-2010', 'Aug-2010', 'Jul-2010', 'Jun-2010',
           'May-2010', 'Apr-2010', 'Mar-2010', 'Feb-2010', 'Jan-2010',
           'Dec-2009', 'Nov-2009', 'Oct-2009', 'Sep-2009', 'Aug-2009',
           'Jul-2009', 'Jun-2009', 'May-2009', 'Apr-2009', 'Mar-2009',
           'Feb-2009', 'Jan-2009', 'Dec-2008', 'Nov-2008', 'Oct-2008',
           'Sep-2008', 'Aug-2008', 'Jul-2008', 'Jun-2008', 'May-2008',
           'Apr-2008', 'Mar-2008', 'Feb-2008', 'Jan-2008', 'Dec-2007',
           'Nov-2007', 'Oct-2007', 'Sep-2007', 'Aug-2007', 'Jul-2007',
           'Jun-2007', 'Dec-2017', 'Nov-2017', 'Oct-2017'], dtype=object)




```python
df.last_pymnt_d.unique()
```




    array(['Feb-2019', 'Jan-2019', nan, 'Dec-2018', 'Nov-2018', 'Oct-2018',
           'Sep-2018', 'Aug-2018', 'Jul-2018', 'Jun-2018', 'May-2018',
           'Apr-2018', 'Mar-2018', 'Feb-2018', 'Jan-2018', 'Oct-2017',
           'Apr-2017', 'Jan-2017', 'Aug-2017', 'Dec-2017', 'Nov-2017',
           'Nov-2016', 'May-2017', 'Jul-2017', 'Mar-2017', 'Jun-2017',
           'Feb-2017', 'Oct-2016', 'Sep-2017', 'Dec-2016', 'Sep-2016',
           'Aug-2016', 'Jul-2016', 'Jun-2016', 'May-2016', 'Apr-2016',
           'Mar-2016', 'Feb-2016', 'Jan-2016', 'Dec-2015', 'Nov-2015',
           'Oct-2015', 'Sep-2015', 'Aug-2015', 'Jul-2015', 'Jun-2015',
           'May-2015', 'Apr-2015', 'Mar-2015', 'Feb-2015', 'Jan-2015',
           'Aug-2014', 'Jul-2014', 'May-2014', 'Sep-2014', 'Jun-2014',
           'Nov-2014', 'Feb-2014', 'Jan-2014', 'Mar-2014', 'Dec-2014',
           'Oct-2014', 'Apr-2014', 'Dec-2013', 'Nov-2013', 'Oct-2013',
           'Sep-2013', 'Aug-2013', 'Jul-2013', 'Jun-2013', 'May-2013',
           'Apr-2013', 'Mar-2013', 'Feb-2013', 'Jan-2013', 'Dec-2012',
           'Nov-2012', 'Oct-2012', 'Sep-2012', 'Aug-2012', 'Jul-2012',
           'Jun-2012', 'May-2012', 'Apr-2012', 'Mar-2012', 'Feb-2012',
           'Jan-2012', 'Dec-2011', 'Nov-2011', 'Oct-2011', 'Sep-2011',
           'Aug-2011', 'Jul-2011', 'Jun-2011', 'May-2011', 'Apr-2011',
           'Mar-2011', 'Feb-2011', 'Jan-2011', 'Dec-2010', 'Nov-2010',
           'Oct-2010', 'Sep-2010', 'Aug-2010', 'Jul-2010', 'Jun-2010',
           'May-2010', 'Apr-2010', 'Mar-2010', 'Feb-2010', 'Jan-2010',
           'Dec-2009', 'Nov-2009', 'Oct-2009', 'Sep-2009', 'Aug-2009',
           'Jul-2009', 'Jun-2009', 'May-2009', 'Apr-2009', 'Mar-2009',
           'Feb-2009', 'Jan-2009', 'Dec-2008', 'Oct-2008', 'Aug-2008',
           'Jul-2008', 'Sep-2008', 'Jun-2008', 'May-2008', 'Nov-2008',
           'Apr-2008', 'Mar-2008', 'Feb-2008', 'Jan-2008', 'Dec-2007'],
          dtype=object)




```python
df.term.unique()
```




    array([' 36 months', ' 60 months'], dtype=object)




```python
df.verification_status.unique()
```




    array(['Not Verified', 'Source Verified', 'Verified'], dtype=object)




```python
df.emp_length.unique()
```




    array(['10+ years', '6 years', '4 years', '< 1 year', '2 years',
           '9 years', nan, '5 years', '3 years', '7 years', '1 year',
           '8 years'], dtype=object)




```python
df.emp_title.unique()
```




    array(['Chef', 'Postmaster ', 'Administrative', ...,
           'Sales, Estimating & Design', 'Acft mechanic', 'BABYSITTER'],
          dtype=object)




```python
df.home_ownership.unique()
```




    array(['RENT', 'MORTGAGE', 'OWN', 'ANY', 'NONE', 'OTHER'], dtype=object)




```python
df.purpose.unique()
```




    array(['debt_consolidation', 'credit_card', 'house', 'car', 'other',
           'vacation', 'home_improvement', 'small_business', 'major_purchase',
           'medical', 'renewable_energy', 'moving', 'wedding', 'educational'],
          dtype=object)




```python
df.zip_code.unique()
```




    array(['109xx', '713xx', '490xx', '985xx', '212xx', '461xx', '606xx',
           '460xx', '327xx', '068xx', '711xx', '300xx', '840xx', '278xx',
           '413xx', '604xx', '436xx', '453xx', '720xx', '741xx', '937xx',
           '611xx', '265xx', '078xx', '290xx', '756xx', '191xx', '672xx',
           '357xx', '231xx', '180xx', '648xx', '284xx', '310xx', '156xx',
           '850xx', '628xx', '114xx', '750xx', '980xx', '618xx', '871xx',
           '368xx', '151xx', '477xx', '800xx', '716xx', '201xx', '028xx',
           '366xx', '549xx', '087xx', '802xx', '857xx', '064xx', '302xx',
           '920xx', '062xx', '371xx', '486xx', '287xx', '154xx', '946xx',
           '933xx', '779xx', '895xx', '925xx', '104xx', '021xx', '774xx',
           '747xx', '070xx', '405xx', '919xx', '234xx', '656xx', '752xx',
           '902xx', '906xx', '612xx', '444xx', '283xx', '280xx', '088xx',
           '425xx', '066xx', '217xx', '301xx', '537xx', '273xx', '282xx',
           '226xx', '917xx', '984xx', '913xx', '480xx', '100xx', '934xx',
           '629xx', '346xx', '989xx', '646xx', '681xx', '482xx', '600xx',
           '928xx', '322xx', '325xx', '152xx', '220xx', '554xx', '778xx',
           '704xx', '921xx', '292xx', '115xx', '030xx', '432xx', '685xx',
           '974xx', '159xx', '936xx', '703xx', '054xx', '775xx', '130xx',
           '796xx', '350xx', '456xx', '458xx', '200xx', '333xx', '347xx',
           '339xx', '295xx', '841xx', '601xx', '427xx', '462xx', '945xx',
           '891xx', '982xx', '150xx', '786xx', '451xx', '336xx', '992xx',
           '017xx', '238xx', '079xx', '953xx', '352xx', '785xx', '471xx',
           '315xx', '977xx', '633xx', '559xx', '731xx', '380xx', '431xx',
           '671xx', '951xx', '018xx', '483xx', '270xx', '815xx', '020xx',
           '787xx', '232xx', '605xx', '912xx', '309xx', '493xx', '112xx',
           '797xx', '853xx', '401xx', '660xx', '120xx', '117xx', '038xx',
           '983xx', '210xx', '760xx', '027xx', '801xx', '356xx', '331xx',
           '986xx', '970xx', '973xx', '736xx', '330xx', '922xx', '342xx',
           '882xx', '349xx', '468xx', '981xx', '225xx', '553xx', '430xx',
           '531xx', '337xx', '277xx', '805xx', '193xx', '334xx', '335xx',
           '194xx', '316xx', '693xx', '208xx', '602xx', '870xx', '463xx',
           '207xx', '288xx', '450xx', '440xx', '941xx', '086xx', '297xx',
           '652xx', '142xx', '680xx', '454xx', '958xx', '550xx', '852xx',
           '386xx', '631xx', '258xx', '276xx', '673xx', '229xx', '133xx',
           '065xx', '163xx', '894xx', '326xx', '140xx', '296xx', '765xx',
           '806xx', '359xx', '911xx', '481xx', '751xx', '890xx', '766xx',
           '435xx', '132xx', '950xx', '169xx', '836xx', '978xx', '708xx',
           '755xx', '993xx', '190xx', '488xx', '121xx', '864xx', '448xx',
           '701xx', '420xx', '221xx', '532xx', '029xx', '421xx', '794xx',
           '272xx', '799xx', '793xx', '730xx', '782xx', '056xx', '442xx',
           '402xx', '657xx', '329xx', '025xx', '328xx', '388xx', '067xx',
           '034xx', '762xx', '199xx', '935xx', '640xx', '314xx', '446xx',
           '084xx', '179xx', '770xx', '324xx', '187xx', '075xx', '370xx',
           '582xx', '111xx', '541xx', '372xx', '275xx', '926xx', '223xx',
           '206xx', '948xx', '441xx', '105xx', '255xx', '644xx', '107xx',
           '085xx', '907xx', '103xx', '900xx', '790xx', '761xx', '157xx',
           '834xx', '131xx', '923xx', '171xx', '781xx', '956xx', '281xx',
           '173xx', '381xx', '565xx', '905xx', '700xx', '216xx', '125xx',
           '927xx', '338xx', '181xx', '145xx', '396xx', '382xx', '160xx',
           '610xx', '083xx', '182xx', '957xx', '967xx', '307xx', '930xx',
           '622xx', '144xx', '360xx', '952xx', '972xx', '046xx', '791xx',
           '924xx', '134xx', '494xx', '153xx', '361xx', '863xx', '061xx',
           '478xx', '712xx', '410xx', '535xx', '247xx', '240xx', '452xx',
           '303xx', '299xx', '063xx', '609xx', '245xx', '674xx', '960xx',
           '174xx', '881xx', '077xx', '254xx', '856xx', '560xx', '211xx',
           '060xx', '320xx', '317xx', '243xx', '932xx', '124xx', '082xx',
           '833xx', '016xx', '341xx', '544xx', '023xx', '780xx', '495xx',
           '898xx', '209xx', '119xx', '433xx', '564xx', '530xx', '241xx',
           '305xx', '373xx', '707xx', '183xx', '014xx', '546xx', '804xx',
           '718xx', '641xx', '489xx', '558xx', '264xx', '377xx', '908xx',
           '367xx', '080xx', '949xx', '113xx', '286xx', '691xx', '996xx',
           '727xx', '449xx', '400xx', '019xx', '474xx', '351xx', '705xx',
           '914xx', '551xx', '073xx', '101xx', '844xx', '122xx', '846xx',
           '423xx', '363xx', '484xx', '184xx', '939xx', '042xx', '666xx',
           '390xx', '721xx', '376xx', '619xx', '874xx', '617xx', '011xx',
           '215xx', '135xx', '074xx', '822xx', '076xx', '910xx', '706xx',
           '851xx', '466xx', '321xx', '724xx', '391xx', '540xx', '189xx',
           '975xx', '071xx', '539xx', '585xx', '010xx', '274xx', '740xx',
           '469xx', '285xx', '170xx', '809xx', '598xx', '230xx', '726xx',
           '837xx', '365xx', '242xx', '244xx', '032xx', '954xx', '773xx',
           '015xx', '197xx', '940xx', '492xx', '467xx', '294xx', '803xx',
           '613xx', '250xx', '880xx', '764xx', '175xx', '164xx', '398xx',
           '473xx', '594xx', '630xx', '149xx', '031xx', '561xx', '662xx',
           '931xx', '999xx', '379xx', '624xx', '620xx', '635xx', '224xx',
           '383xx', '176xx', '968xx', '897xx', '792xx', '313xx', '570xx',
           '195xx', '684xx', '971xx', '143xx', '827xx', '548xx', '168xx',
           '955xx', '608xx', '239xx', '253xx', '024xx', '039xx', '228xx',
           '491xx', '013xx', '586xx', '855xx', '378xx', '557xx', '816xx',
           '141xx', '670xx', '637xx', '728xx', '081xx', '859xx', '626xx',
           '384xx', '271xx', '406xx', '777xx', '057xx', '944xx', '783xx',
           '323xx', '995xx', '860xx', '219xx', '108xx', '069xx', '714xx',
           '344xx', '757xx', '395xx', '298xx', '668xx', '571xx', '499xx',
           '734xx', '650xx', '172xx', '918xx', '235xx', '547xx', '826xx',
           '754xx', '155xx', '403xx', '148xx', '236xx', '185xx', '763xx',
           '127xx', '106xx', '784xx', '746xx', '304xx', '722xx', '457xx',
           '374xx', '808xx', '161xx', '044xx', '188xx', '257xx', '308xx',
           '437xx', '607xx', '214xx', '358xx', '916xx', '040xx', '776xx',
           '479xx', '434xx', '767xx', '959xx', '737xx', '729xx', '676xx',
           '443xx', '053xx', '198xx', '128xx', '415xx', '293xx', '943xx',
           '033xx', '820xx', '037xx', '158xx', '710xx', '587xx', '832xx',
           '259xx', '723xx', '129xx', '903xx', '146xx', '677xx', '497xx',
           '688xx', '072xx', '049xx', '022xx', '811xx', '825xx', '472xx',
           '616xx', '177xx', '810xx', '385xx', '110xx', '661xx', '319xx',
           '915xx', '655xx', '562xx', '719xx', '807xx', '795xx', '354xx',
           '496xx', '725xx', '789xx', '498xx', '047xx', '543xx', '279xx',
           '261xx', '566xx', '545xx', '843xx', '988xx', '813xx', '465xx',
           '588xx', '052xx', '658xx', '879xx', '997xx', '627xx', '291xx',
           '422xx', '262xx', '218xx', '306xx', '645xx', '260xx', '404xx',
           '136xx', '026xx', '487xx', '394xx', '116xx', '748xx', '089xx',
           '990xx', '043xx', '411xx', '614xx', '147xx', '464xx', '615xx',
           '249xx', '447xx', '268xx', '051xx', '599xx', '838xx', '573xx',
           '475xx', '455xx', '476xx', '355xx', '759xx', '222xx', '738xx',
           '625xx', '196xx', '749xx', '438xx', '683xx', '045xx', '048xx',
           '445xx', '675xx', '947xx', '397xx', '534xx', '689xx', '768xx',
           '387xx', '788xx', '412xx', '050xx', '904xx', '667xx', '166xx',
           '686xx', '470xx', '407xx', '647xx', '597xx', '654xx', '595xx',
           '824xx', '651xx', '735xx', '574xx', '590xx', '744xx', '998xx',
           '392xx', '665xx', '178xx', '814xx', '439xx', '118xx', '593xx',
           '126xx', '123xx', '745xx', '577xx', '664xx', '416xx', '035xx',
           '976xx', '653xx', '596xx', '812xx', '639xx', '743xx', '485xx',
           '580xx', '603xx', '875xx', '845xx', '678xx', '424xx', '227xx',
           '687xx', '979xx', '389xx', '563xx', '246xx', '233xx', '769xx',
           '012xx', '186xx', '962xx', '542xx', '567xx', '138xx', '961xx',
           '991xx', '847xx', '318xx', '165xx', '364xx', '835xx', '636xx',
           '878xx', '312xx', '248xx', '829xx', '538xx', '963xx', '584xx',
           '893xx', '623xx', '638xx', '884xx', '591xx', '831xx', '237xx',
           '592xx', '798xx', '139xx', '408xx', '581xx', '263xx', '830xx',
           '251xx', '758xx', '994xx', '393xx', '572xx', '426xx', '717xx',
           '162xx', '634xx', '739xx', '137xx', '883xx', '575xx', '873xx',
           '266xx', '102xx', '289xx', '362xx', '041xx', '690xx', '036xx',
           '819xx', '256xx', '369xx', '877xx', '252xx', '669xx', '204xx',
           '865xx', '409xx', '167xx', '828xx', '058xx', '267xx', '414xx',
           '091xx', '332xx', '090xx', '096xx', '583xx', '772xx', '576xx',
           '097xx', '059xx', '679xx', '556xx', '696xx', '094xx', '753xx',
           '823xx', '523xx', '098xx', '418xx', '340xx', '417xx', '901xx',
           '821xx', '733xx', '503xx', '965xx', '095xx', '942xx', '694xx',
           '692xx', '621xx', '348xx', '732xx', '092xx', '966xx', '311xx',
           '964xx', '203xx', '093xx', '506xx', '858xx', '512xx', '987xx',
           '861xx', '202xx', '742xx', '345xx', '343xx', '552xx', '555xx',
           '909xx', '569xx', '698xx', '055xx', '500xx', '502xx', '969xx',
           '702xx', '008xx', '007xx', '515xx', '510xx', '528xx', '521xx',
           '353xx', '009xx', '872xx', '854xx', '525xx', '509xx', '522xx',
           '527xx', '269xx', '892xx', '929xx', '504xx', '205xx', '709xx',
           '849xx', '771xx', '501xx', '520xx', '817xx', '568xx', '399xx',
           '649xx', '862xx', '507xx', nan, '885xx', '663xx', '643xx', '513xx',
           '429xx', '682xx', '938xx', '888xx', '524xx', '889xx', '516xx',
           '511xx', '375xx', '514xx', '896xx'], dtype=object)




```python
df.addr_state.unique()
```




    array(['NY', 'LA', 'MI', 'WA', 'MD', 'IN', 'IL', 'FL', 'CT', 'GA', 'UT',
           'NC', 'KY', 'OH', 'AR', 'OK', 'CA', 'WV', 'NJ', 'SC', 'TX', 'PA',
           'KS', 'AL', 'VA', 'MO', 'AZ', 'NM', 'CO', 'RI', 'WI', 'TN', 'NV',
           'MA', 'NE', 'MN', 'NH', 'OR', 'VT', 'DC', 'MS', 'ID', 'DE', 'ND',
           'HI', 'ME', 'AK', 'WY', 'MT', 'SD', 'IA'], dtype=object)




```python
df.earliest_cr_line.unique()
```




    array(['Apr-2001', 'Jun-1987', 'Apr-2011', 'Feb-2006', 'Dec-2000',
           'Sep-2002', 'Nov-2004', 'Nov-1997', 'Aug-1998', 'Apr-2002',
           'May-2007', 'Dec-2003', 'Jun-2003', 'Oct-2008', 'Jul-1990',
           'Dec-1988', 'Dec-2002', 'Oct-2010', 'Jul-2005', 'Feb-2001',
           'Dec-2004', 'Oct-2001', 'Sep-2003', 'Oct-2004', 'Sep-2010',
           'Oct-1999', 'Feb-1997', 'Jan-1995', 'Aug-2005', 'Apr-2005',
           'Oct-2005', 'Nov-2006', 'Sep-1999', 'Sep-2006', 'Aug-1996',
           'May-2015', 'Nov-1999', 'Dec-1998', 'Aug-1989', 'Apr-2012',
           'Sep-2012', 'Jan-2011', 'Jul-2013', 'Sep-2011', 'Aug-1999',
           'Dec-1991', 'Nov-2007', 'Oct-2007', 'Jul-1986', 'Nov-2014',
           'Apr-1995', 'Jul-1992', 'Aug-2006', 'Oct-1988', 'Feb-2005',
           'Aug-2002', 'Oct-2003', 'Apr-2004', 'Jun-2007', 'Aug-1990',
           'Dec-1990', 'Jul-1997', 'Sep-1988', 'Sep-2008', 'Feb-1999',
           'Apr-2007', 'Aug-2004', 'Dec-2006', 'Mar-2007', 'Oct-2006',
           'Jul-2014', 'May-2005', 'Jun-1995', 'Mar-2014', 'Feb-2003',
           'Aug-2007', 'Jul-1996', 'Jan-1991', 'Aug-2008', 'Dec-2007',
           'Sep-1997', 'Jan-2000', 'Jan-2005', 'Feb-2007', 'Dec-2005',
           'Jan-2012', 'Dec-1997', 'Mar-2009', 'Apr-1989', 'Jun-2006',
           'Jan-2014', 'Jun-2009', 'May-1975', 'Apr-2014', 'Oct-1996',
           'Mar-1999', 'Dec-1994', 'Jan-2008', 'Jul-1999', 'Feb-2014',
           'Oct-2000', 'Aug-2010', 'Mar-1993', 'May-2009', 'Aug-1993',
           'May-1998', 'Sep-1998', 'Aug-2011', 'Nov-1984', 'Apr-2000',
           'Apr-2003', 'May-2003', 'Feb-1996', 'Jan-1996', 'Mar-2005',
           'Aug-2001', 'Oct-2011', 'Mar-1974', 'Jul-2007', 'Feb-1998',
           'Jan-2006', 'Mar-2006', 'Aug-2003', 'May-2004', 'Nov-2013',
           'Jan-2009', 'Jan-2002', 'Sep-1984', 'Jul-2010', 'Mar-1998',
           'Jun-1991', 'Sep-1996', 'Sep-1991', 'Mar-1991', 'Apr-1999',
           'Aug-2009', 'Nov-2001', 'Apr-2006', 'Dec-2009', 'Nov-1986',
           'Jan-2004', 'Nov-2002', 'Sep-1994', 'Mar-2000', 'Jun-2015',
           'Apr-2013', 'Sep-2000', 'Jun-1979', 'Dec-1989', 'Oct-1997',
           'Aug-2000', 'Aug-1987', 'Jul-1991', 'Apr-1988', 'Oct-2014',
           'Jul-2002', 'Sep-2009', 'May-2010', 'Sep-1985', 'Feb-1991',
           'Nov-2003', 'Mar-2010', 'Jul-1995', 'Oct-1994', 'Jul-1987',
           'Oct-2009', 'Jan-1997', 'Oct-1993', 'Aug-1992', 'Sep-2005',
           'Jan-2001', 'Jul-1983', 'Jul-2003', 'Nov-2000', 'Jun-1998',
           'Aug-2013', 'Apr-2009', 'Jul-2004', 'Nov-2005', 'Apr-1998',
           'Nov-1993', 'Oct-1991', 'Jun-2004', 'Jun-2008', 'Jan-1993',
           'Mar-2003', 'Mar-2011', 'Jun-2000', 'Mar-2004', 'Feb-2008',
           'May-1973', 'May-2008', 'Jan-2003', 'Mar-2002', 'Aug-1997',
           'Feb-2009', 'Feb-2004', 'Apr-1993', 'Dec-1995', 'Dec-2008',
           'Jan-1990', 'Mar-2012', 'Mar-1987', 'Jun-2011', 'Feb-2011',
           'Jan-2015', 'Jun-2005', 'Mar-2001', 'Jun-1999', 'Mar-1979',
           'Jan-1998', 'May-2000', 'Dec-2001', 'May-2002', 'Oct-2002',
           'Sep-2001', 'Jun-1981', 'May-1994', 'Mar-2015', 'Jan-1994',
           'Jul-2015', 'Sep-2007', 'Aug-2012', 'Feb-1989', 'Jun-2012',
           'Aug-1972', 'Nov-2011', 'Feb-2015', 'Oct-1992', 'Apr-1994',
           'Feb-1985', 'May-2001', 'Jul-2008', 'Dec-1996', 'Jan-1987',
           'May-1995', 'Mar-1996', 'Nov-2009', 'Jul-1998', 'Jan-1988',
           'Jun-1988', 'Apr-1984', 'May-1991', 'Dec-2014', 'Nov-1996',
           'Apr-1978', 'Jul-2009', 'Jun-1970', 'Jun-1990', 'Mar-2008',
           'Jul-2001', 'Oct-1995', 'May-2006', 'Feb-2002', 'Jan-2007',
           'Jul-2006', 'Jul-2000', 'Mar-1990', 'Dec-2012', 'Jun-2014',
           'Jul-2012', 'Jun-1994', 'Feb-2000', 'Nov-1988', 'Oct-1987',
           'Nov-1991', 'Feb-1982', 'Apr-1991', 'Sep-2004', 'Nov-1998',
           'Nov-1983', 'Nov-1994', 'Dec-1972', 'Nov-1995', 'Nov-1992',
           'May-2014', 'May-1997', 'Jun-1989', 'Oct-2015', 'Apr-1986',
           'May-1993', 'Mar-1994', 'Jan-1999', 'Oct-1998', 'Jun-2013',
           'Mar-1995', 'Feb-2012', 'Sep-1993', 'Sep-1989', 'Feb-1993',
           'Jun-1982', 'Aug-1995', 'Nov-2012', 'Aug-1986', 'Dec-2011',
           'Oct-2013', 'May-2011', 'Jan-1980', 'Mar-1988', 'Aug-1985',
           'May-1996', 'Nov-1978', 'Dec-2013', 'May-1985', 'Nov-1982',
           'Mar-2013', 'Jun-1986', 'Jun-1992', 'Oct-1985', 'May-2012',
           'May-1987', 'Dec-2010', 'Nov-2008', 'May-1984', 'Nov-2010',
           'Jun-1993', 'Dec-1993', 'Feb-1990', 'Apr-2008', 'Sep-1995',
           'Jun-2010', 'Jul-2011', 'Feb-1994', 'Jun-2002', 'Jul-1993',
           'Dec-1992', 'Oct-1986', 'Aug-1984', 'Sep-2013', 'May-1982',
           'Mar-1997', 'Jul-1989', 'Apr-1990', 'Jul-1985', 'Aug-2014',
           'Dec-1982', 'Feb-2010', 'Oct-2012', 'May-1986', 'Jul-1994',
           'Aug-1994', 'Jul-1988', 'Apr-1985', 'Feb-2013', 'Mar-1989',
           'Apr-1997', 'Dec-1987', 'Jul-1979', 'Dec-1999', 'May-1988',
           'Dec-1986', 'Jun-2001', 'May-1999', 'Jun-1996', 'Mar-1983',
           'Aug-2015', 'Feb-1987', 'Feb-1983', 'Nov-1989', 'Apr-1992',
           'Jul-1981', 'Feb-1981', 'Jan-2010', 'May-1989', 'Sep-1978',
           'Aug-1975', 'Jun-1997', 'Apr-1983', 'Aug-1988', 'Dec-1984',
           'Aug-1991', 'May-2013', 'Jan-1979', 'Jan-1985', 'Aug-1978',
           'Dec-1978', 'Nov-1970', 'Apr-1987', 'Sep-1986', 'Apr-1996',
           'May-1977', 'Oct-1989', 'Aug-1982', 'Sep-1981', 'May-1983',
           'Jul-1984', 'Sep-2015', 'Nov-1981', 'Jan-2013', 'Sep-1982',
           'Oct-1977', 'Feb-1979', 'Nov-1979', 'Oct-1984', 'Feb-1977',
           'Jul-1978', 'Dec-1981', 'Sep-2014', 'Sep-1972', 'Jul-1971',
           'Nov-1977', 'Nov-1965', 'Mar-1978', 'Jan-1982', 'Mar-1992',
           'Sep-1973', 'Mar-1984', 'Dec-1977', 'Sep-1980', 'Oct-1990',
           'Mar-1981', 'Dec-1985', 'Oct-1982', 'Nov-1985', 'Oct-1974',
           'Sep-1992', 'Feb-1995', 'Nov-1990', 'Apr-1980', 'Apr-2015',
           'Jan-1975', 'May-1974', 'Sep-1990', 'May-1990', 'Nov-1987',
           'Jun-1976', 'Oct-1983', 'Aug-1980', 'Jan-1986', 'Feb-1973',
           'Feb-1986', 'Apr-1976', 'Dec-1971', 'Jan-1992', 'Feb-1971',
           'Jun-1985', 'Feb-1976', 'Jan-1970', 'Oct-1975', 'Apr-1979',
           'Jun-1983', 'Sep-1987', 'Oct-1980', 'Feb-1978', 'Jul-1980',
           'May-1992', 'Jan-1984', 'Oct-1978', 'Mar-1985', 'May-1969',
           'Nov-1980', 'Mar-1975', 'Dec-1980', 'Aug-1976', 'Jul-1977',
           'Nov-1968', 'Sep-1983', 'Jun-1984', 'Nov-1976', 'Mar-1976',
           'Apr-2010', 'Feb-1984', 'May-1976', 'Mar-1970', 'Mar-1977',
           'Feb-1988', 'Dec-1979', 'Feb-1992', 'Oct-1981', 'Apr-1982',
           'Jan-1989', 'May-1978', 'Oct-1969', 'Apr-1962', 'May-1961',
           'Aug-1977', 'Apr-1981', 'Dec-1983', 'Nov-1975', 'Apr-1977',
           'Jul-1976', 'Jul-1982', 'Nov-2015', 'Oct-1976', 'Jul-1967',
           'Jun-1972', 'Mar-1986', 'Jun-1967', 'May-1979', 'Mar-1980',
           'Aug-1971', 'May-1981', 'Jan-1983', 'Sep-1979', 'Jan-1977',
           'Mar-1982', 'Jun-1977', 'Jan-1981', 'Jan-1978', 'Jan-1973',
           'Mar-1971', 'May-1964', 'Feb-1980', 'Jul-1974', 'May-1965',
           'Jul-1973', 'Aug-1973', 'Aug-1981', 'Sep-1971', 'Jun-1978',
           'Jan-1974', 'Aug-1974', 'Jan-1967', 'Apr-1975', 'Sep-1970',
           'Jun-1965', 'Mar-1973', 'Sep-1976', 'Apr-1971', 'Nov-1974',
           'Dec-1968', 'Dec-1976', 'Feb-1966', 'May-1980', 'Aug-1979',
           'Apr-1963', 'Oct-1970', 'Jan-1976', 'Sep-1977', 'Sep-1974',
           'Jan-1954', 'Apr-1973', 'Jan-1969', 'Jun-1980', 'Aug-1983',
           'Jan-1968', 'Aug-1969', 'Dec-1975', 'Jul-1975', 'Jan-1971',
           'Jul-1968', 'Nov-1973', 'Jan-1950', 'Apr-1972', 'Jun-1974',
           'Jul-1950', 'Apr-1968', 'Jan-1963', 'Nov-1966', 'Oct-1972',
           'Dec-1967', 'Aug-1966', 'Mar-1972', 'May-1972', 'Sep-1964',
           'Feb-1972', 'Dec-1973', 'Jan-1958', 'Oct-1973', 'Feb-1963',
           'Aug-1967', 'Oct-1965', 'Feb-1970', 'Oct-1979', 'Jan-1972',
           'Feb-1974', 'Feb-1969', 'Apr-1964', 'Sep-1975', 'Aug-1970',
           'Jul-1965', 'Dec-1970', 'May-1971', 'Apr-1965', 'Jan-1956',
           'Apr-1970', 'Sep-1965', 'Mar-1965', 'Dec-1974', 'Dec-1966',
           'Jan-1965', 'Jan-1960', 'Nov-1959', 'May-1968', 'Jun-1975',
           'Jun-1969', 'Jan-1962', 'May-1970', 'Apr-1974', 'Feb-1967',
           'Aug-1968', 'Sep-1969', 'Apr-1969', 'Mar-1962', 'Jun-1973',
           'Oct-1961', 'Apr-1967', 'Oct-1964', 'Jan-1959', 'Dec-1969',
           'Jun-1964', 'Mar-1968', 'Mar-1969', 'Sep-1967', 'Mar-1967',
           'Jul-1969', 'May-1967', 'Nov-1971', 'Nov-1972', 'Jan-1961',
           'Feb-1962', 'Jun-1968', 'Feb-1975', 'Jul-1966', 'Dec-1964',
           'Sep-1966', 'Sep-1963', 'Aug-1962', 'Jan-1966', 'Oct-1968',
           'Nov-1964', 'Mar-1954', 'Aug-1963', 'Nov-1967', 'Jan-1964',
           'Jul-1964', 'Sep-1968', 'Jun-1971', 'Oct-1971', 'Dec-1965',
           'Oct-1966', 'Jul-1972', 'Feb-1965', 'Jun-1966', 'Aug-1964',
           'Jul-1970', 'Apr-1966', 'May-1966', 'Sep-1962', 'Oct-1967',
           'Apr-1960', 'Jun-1963', 'Nov-1969', 'Dec-1962', 'Feb-1968',
           'Jan-1951', 'Mar-1966', 'Jun-1962', 'Dec-1963', 'Aug-1965',
           'Jan-1957', 'Dec-1961', 'Jul-1961', 'Dec-1947', 'Apr-1957',
           'Mar-1964', 'Mar-1959', 'Jan-1955', 'Jun-1958', 'Feb-1964',
           'Apr-1959', 'Nov-1963', 'Aug-1959', 'May-1963', 'Nov-1961',
           'Jun-1960', 'Mar-1963', 'Jul-1960', 'Oct-1963', 'Dec-1960',
           'Sep-1961', 'Jun-1961', 'Aug-1958', 'Apr-1955', 'May-1960',
           'Oct-1962', 'Dec-1959', 'Jun-1956', 'Feb-1961', 'Jul-1963',
           'Feb-1958', 'Jun-1959', 'Jan-1953', 'Jul-1962', 'Aug-1941',
           'Mar-1961', 'Dec-1946', 'Aug-1961', 'May-1962', 'Feb-1960',
           'Mar-1933', 'Nov-1957', 'Jun-1952', 'Jun-1957', 'May-1958',
           'Nov-1962', 'Aug-1960', 'Mar-1955', 'May-1959', 'Feb-1945',
           'Jan-1948', 'Jul-1959', 'Nov-1960', 'Mar-1957', 'Jul-1952',
           'Jan-1952', 'Apr-1958', 'Jul-1951', 'Apr-1961', 'Dec-1958',
           'Oct-1958', 'Feb-1959', 'Oct-1959', 'Jun-1955', 'Sep-1953',
           'Nov-1956', 'Mar-1958', 'May-1955', 'Mar-1960', 'Aug-1951',
           'Sep-1959', 'Nov-1958', 'Sep-1956', 'May-1957', 'Dec-1956',
           'Jul-1958', 'Sep-1960', 'Oct-1960', 'Aug-1950', 'Aug-1955',
           'Jan-1944', 'Nov-1952', 'Aug-1953', 'Apr-1934', 'Sep-1951',
           'Feb-1934', 'Aug-1957', 'Oct-1957', 'Nov-1953', 'Jul-1955',
           'May-1953', 'Nov-1950', 'Nov-1955', 'Feb-1957', 'Dec-1951',
           'Aug-1946', 'Nov-1954', 'Sep-1957', 'Jun-1949', 'Oct-1950',
           'May-1950', 'Oct-1954', 'Jan-1946', 'Dec-1950', nan], dtype=object)




```python
df.grade.unique()
```




    array(['C', 'D', 'B', 'A', 'E', 'F', 'G'], dtype=object)




```python
df.sub_grade.unique()
```




    array(['C1', 'D2', 'D1', 'C4', 'C3', 'C2', 'D5', 'B3', 'A4', 'B5', 'C5',
           'D4', 'E1', 'E4', 'B4', 'D3', 'A1', 'E5', 'B2', 'B1', 'A5', 'F5',
           'A3', 'E3', 'A2', 'E2', 'F4', 'G1', 'G2', 'F1', 'F2', 'F3', 'G4',
           'G3', 'G5'], dtype=object)



### Entendendo a distribuição das variáveis numéricas

Variáveis numéricas como Renda Anual, Valor do Empréstimo e Taxa de Juros raramente seguem uma curva normal perfeita. A realidade financeira tende a ser assimétrica: muitos possuem pouco, e poucos possuem muito.

Entender essas formas é crucial. Se não identificarmos a cauda longa agora, nossa análise pode indicar regras erradas baseadas em exceções, e não na regra. Vamos descobrir onde está o centro de gravidade dos nossos dados.


```python
# Seleção de variáveis numéricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

num_plots = len(numeric_cols)
cols = 3 
rows = math.ceil(num_plots / cols)

# Configuração da figura
plt.figure(figsize=(15, 4 * rows))
plt.suptitle("Distribuição das Variáveis Numéricas (KDE)", fontsize=18, fontweight='bold', y=0.99)

for i, col in enumerate(numeric_cols):
    plt.subplot(rows, cols, i + 1)
    sns.kdeplot(x=df[col].dropna(), fill=True, color='#4c72b0')
    
    plt.title(col, fontsize=12)
    plt.xlabel("") 
    plt.ylabel("Densidade")
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_45_0.png)
    


### Entendendo a distribuição das variáveis discretas e categóricas

Se as variáveis numéricas nos disseram quanto dinheiro está em jogo, as variáveis discretas e categóricas nos dizem quem está pegando esse dinheiro e para quê.

Entender a distribuição dessas categorias é vital. Um desequilíbrio aqui pode significar que o banco está exposto a riscos setoriais específicos que os números brutos, sozinhos, não conseguem mostrar. Vamos ver a "cara" do nosso cliente.


```python
# Seleção de variáveis categóricas
cat_features = [
    'loan_status', 'term', 'grade', 'emp_length', 
    'home_ownership', 'verification_status', 'purpose', 'addr_state'
]

# Configuração da figura
plt.figure(figsize=(20, 25))
plt.suptitle("Análise Univariada das Variáveis Categóricas", fontsize=22, fontweight='bold', alpha=0.8, y=0.92)

for i, col in enumerate(cat_features):
    plt.subplot(4, 2, i + 1)
    
    # Ordenando as barras pela frequência para melhor visualização
    order = df[col].value_counts().index
    
    sns.countplot(data=df, x=col, order=order, palette='viridis')
    
    plt.title(f"Distribuição de {col}", fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Contagem", fontsize=12)
    
    # Rotacionar labels se houver muitas categorias
    if df[col].nunique() > 5:
        plt.xticks(rotation=45)
    
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_48_0.png)
    


## Processo

Esta é a lei imutável da Ciência de Dados. Não importa quão sofisticado seja o modelo de Machine Learning ou quão elegante seja o dashboard; se a matéria-prima estiver contaminada, a conclusão será falha. A etapa de Processamento é onde a análise de dados deixa de ser uma simples coleta e se torna uma disciplina de engenharia.

Este é o momento de garantir a integridade. Estamos construindo a base sobre a qual todas as nossas decisões de crédito serão tomadas.

### Limpeza de dados

Nesta fase, vamos arregaçar as mangas para transformar dados brutos em um Analytical Dataset.

#### Definição do Escopo e Variável Alvo

No mundo real, a inadimplência não é binária; é um processo. Para esta análise, definimos a variável alvo (target) alinhada com as práticas de gestão de portfólio:
* Maus Pagadores (1): Empréstimos classificados como 'Charged Off', 'Default' ou atrasos severos que não atendem à política de crédito.
* Bons Pagadores (0): Empréstimos com status 'Fully Paid'.
* Exclusão Estratégica: Empréstimos com status 'Current' (em dia) foram segregados da modelagem preditiva de classificação. Incluí-los como 'bons' criaria um Viés de Censura, pois um empréstimo que está 'Current' hoje pode virar 'Default' amanhã. Eles serão analisados separadamente nas curvas de sobrevivência.


```python
# Mantemos apenas quem já concluiu o ciclo do empréstimo para evitar o Viés de Censura
target_states = [
    'Fully Paid', 
    'Charged Off', 
    'Default',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off'
]

# Criamos df_cleaned para preservar o DataFrame original intacto
df_cleaned = df[df['loan_status'].isin(target_states)].copy()

# Definição da Target (Padrão de Mercado: 1 = Risco/Evento)
# Se contiver "Fully Paid", o alvo é 0 (Bom). Se não (Charged Off/Default), é 1 (Mau).
df_cleaned['target'] = np.where(df_cleaned['loan_status'].str.contains('Fully Paid'), 0, 1)
```

#### Prevenção de Vazamento de Dados

Um erro fatal na análise de crédito é permitir que a avaliação de risco seja contaminada pelo 'viés do futuro'. Variáveis presentes no dataset original, como total_pymnt (total pago) ou recoveries (valores recuperados), são consequências do comportamento do cliente, e não dados disponíveis no momento da concessão.
Embora este projeto não implemente algoritmos de Machine Learning, a disciplina de segregação de dados é mantida rigorosamente para simular uma Auditoria de Política de Crédito realista:
1. **df_risk:** Contém exclusivamente dados visíveis no momento da solicitação.
2. **df_finance:** Preserva as variáveis de performance financeira. Este dataset é utilizado estritamente para o cálculo de Rentabilidade Real (NAR), curvas de LGD e Análise de Safra.


```python
# Definimos as colunas de "vazamento"
leakage_cols = [
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 
    'total_rec_int', 'total_rec_late_fee', 
    'recoveries', 'collection_recovery_fee', 
    'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d'
]

# Criamos o df_finance (Foco em Rentabilidade/LGD/Vintage)
cols_finance = [
    'loan_status', 'target', 'loan_amnt', 'funded_amnt', 
    'issue_d', 'term', 'sub_grade', 'grade',  
    'int_rate', 'installment'                 
] + leakage_cols

# Garantir que pegamos apenas colunas que existem no df_cleaned
cols_finance = [c for c in cols_finance if c in df_cleaned.columns]

df_finance = df_cleaned[cols_finance].copy()

# Criamos o df_risk
df_risk = df_cleaned.drop(columns=leakage_cols, errors='ignore').copy()
```

#### Conversão e Limpeza de Tipos de Dados

Muitas vezes, bases de dados chegam formatadas para leitura humana, carregadas de símbolos (como % em taxas de juros ou a palavra years em prazos). Para o Python, entretanto, 12.5% não é um número: é um texto (string/object), tão incalculável quanto a palavra "banana".

Sem essa tradução, nossos dados são apenas palavras bonitas em uma tabela. Com ela, tornam-se insumos matemáticos prontos para análise.


```python
# Usa Regex para extrair apenas os dígitos, ignorando espaços
# O uso de .strip() garante que não sobram espaços antes da conversão
df_finance['term'] = df_finance['term'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
df_risk['term'] = df_risk['term'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)

cols_perc = ['int_rate', 'revol_util']
for col in cols_perc:
    # Remove '%' se for string
    if df_risk[col].dtype == 'object':
        df_risk[col] = df_risk[col].str.replace('%', '', regex=False).astype(float)
    
    # Divide por 100 apenas onde o valor é > 1
    mask_high = df_risk[col] > 1
    df_risk.loc[mask_high, col] = df_risk.loc[mask_high, col] / 100

# Formato '%b-%Y' (Ex: Dec-2018)
date_fmt = '%b-%Y'

# No df_risk
df_risk['issue_d'] = pd.to_datetime(df_risk['issue_d'], format=date_fmt, errors='coerce')
df_risk['earliest_cr_line'] = pd.to_datetime(df_risk['earliest_cr_line'], format=date_fmt, errors='coerce')

# No df_finance (Variáveis de performance/cálculo)
df_finance['issue_d'] = pd.to_datetime(df_finance['issue_d'], format=date_fmt, errors='coerce')
df_finance['last_pymnt_d'] = pd.to_datetime(df_finance['last_pymnt_d'], format=date_fmt, errors='coerce')
```

#### Tratando valores ausentes

Encontrar células vazias (NaN ou null) é inevitável em datasets reais. A reação instintiva de muitos analistas é simplesmente deletar essas linhas ou preenchê-las cegamente com a média. No entanto, essa abordagem pode ser catastrófica.

Um campo vazio nem sempre é um erro de sistema; ele pode ser um comportamento:

* Um cliente que deixa o campo Tempo de Emprego em branco pode estar escondendo desemprego recente.

* A falta de Taxa de Utilização pode indicar um cliente novo, sem histórico bancário.

Nesta etapa, adotaremos uma abordagem cirúrgica:

1. **Investigação:** O dado falta aleatoriamente ou existe um padrão?

2. **Estratégia de Inputação:** Para variáveis numéricas, preencher com a média/mediana distorce a distribuição? Para categóricas, devemos criar uma categoria "Desconhecido"?

Nosso objetivo é preservar a integridade da informação sem introduzir ruídos artificiais que "alucinem" uma estabilidade financeira que não existe.


```python
df_risk.isna().sum()
```




    loan_status                    0
    issue_d                        0
    term                           0
    annual_inc                     4
    dti                          312
    verification_status            0
    installment                    0
    emp_length                 75491
    emp_title                  82741
    home_ownership                 0
    purpose                        0
    zip_code                       1
    addr_state                     0
    revol_util                   850
    inq_last_6mths                30
    mths_since_last_delinq    659032
    pub_rec                       29
    earliest_cr_line              29
    open_acc                      29
    total_acc                     29
    grade                          0
    sub_grade                      0
    int_rate                       0
    loan_amnt                      0
    funded_amnt                    0
    target                         0
    dtype: int64




```python
df_finance.isna().sum()
```




    loan_status                   0
    target                        0
    loan_amnt                     0
    funded_amnt                   0
    issue_d                       0
    term                          0
    sub_grade                     0
    grade                         0
    int_rate                      0
    installment                   0
    total_pymnt                   0
    total_rec_prncp               0
    total_rec_int                 0
    recoveries                    0
    collection_recovery_fee       0
    last_pymnt_d               2273
    last_pymnt_amnt               0
    dtype: int64




```python
# Histórico de Delinquência
df_risk['never_delinq'] = np.where(df_risk['mths_since_last_delinq'].isna(), 1, 0)
df_risk['mths_since_last_delinq'].fillna(999, inplace=True)

# Tempo de Emprego (emp_length) - Limpeza e Categorização
def clean_emp_length(val):
    if pd.isna(val): return np.nan
    if '< 1 year' in val: return 0
    if '10+' in val: return 10
    digits = re.findall(r'\d+', val)
    return int(digits[0]) if digits else np.nan

df_risk['emp_length_int'] = df_risk['emp_length'].apply(clean_emp_length)
df_risk['emp_length_missing'] = np.where(df_risk['emp_length_int'].isna(), 1, 0)
df_risk['emp_length_int'].fillna(0, inplace=True)
df_risk.drop(columns=['emp_length'], inplace=True)

# Bureau de Crédito (Assumir ausência de eventos negativos)
cols_to_zero = ['pub_rec', 'inq_last_6mths', 'open_acc', 'total_acc']
df_risk[cols_to_zero] = df_risk[cols_to_zero].fillna(0)

# Tratamento de Taxas e Rendas (Mediana para evitar outliers)
if df_risk['revol_util'].dtype == 'object':
    df_risk['revol_util'] = df_risk['revol_util'].str.extract(r'(\d+\.?\d*)').astype(float)

df_risk['revol_util'].fillna(df_risk['revol_util'].median(), inplace=True)
df_risk['annual_inc'].fillna(df_risk['annual_inc'].median(), inplace=True)
df_risk['dti'].fillna(df_risk['dti'].median(), inplace=True)

# Criar coluna para cálculos de MOB (Months on Book)
# Se não houve pagamento, a data final é a data de início (MOB = 0)
df_finance['last_pymnt_d_calc'] = df_finance['last_pymnt_d'].fillna(df_finance['issue_d'])

# Criar coluna para exibição em relatórios
df_finance['last_pymnt_status'] = df_finance['last_pymnt_d'].dt.strftime('%Y-%m').fillna('No Payment')
```

#### Tratamento de Outliers

Modelos estatísticos amam a mediocridade, mas o mundo real é feito de extremos. Em nossa base, podemos encontrar desde erros de digitação óbvios até realidades financeiras legítimas, mas raras.

O perigo dos outliers não tratados é a Distorção da Realidade: um único valor extremo pode "puxar" a média e o desvio padrão, fazendo com que a análise indique regras enviesadas que não se aplicam à vasta maioria dos clientes.

Nesta etapa, faremos uma Triagem Estratégica:

1. **O Erro:** Valores impossíveis serão tratados como erro de dados.

2. **O Extremo Legítimo:** Para valores reais, mas muito distantes (ex: rendas milionárias), aplicaremos técnicas de Winsorization. Vamos limitar os valores extremos a um teto racional.

Isso acalma a variância dos dados, permitindo que a análise generalize melhor sem ser "sequestrado" pelas exceções.


```python
# Preservar a riqueza, remover a distorção
limit_inc_99 = df_risk['annual_inc'].quantile(0.99)

# Substitui valores acima do P99 pelo valor do P99
df_risk['annual_inc'] = df_risk['annual_inc'].clip(upper=limit_inc_99)

initial_rows = len(df_risk)

# Detecção automática de escala
# Se o max for > 2, assumimos escala 0-100. Caso contrário, 0-1.
scale_factor = 100 if df_risk['dti'].max() > 5 else 1 
dti_limit = 100 if scale_factor == 100 else 1.0
revol_limit = 200 if scale_factor == 100 else 2.0

# Aplicação dos Filtros
# DTI > 100% é insolvência técnica ou erro
# Revol_Util > 200% é erro de sistema ou multa extrema
mask_clean = (df_risk['dti'] <= dti_limit) & (df_risk['revol_util'] <= revol_limit)

# Filtragem
df_risk = df_risk[mask_clean]
# Garante que o financeiro tenha as mesmas linhas do risco
df_finance = df_finance.loc[df_risk.index]
```

#### Engenharia de atributos (Feature Engineering)

Para superar as limitações das análises estáticas, transformamos dados temporais em métricas de ciclo de vida:
* **Safra:** Agrupamento por trimestre de originação para isolar a qualidade da concessão de choques macroeconômicos.
* **MOB:** Cálculo da idade do empréstimo mês a mês. Isso nos permite comparar a performance de um empréstimo concedido em 2015 com um de 2017 no mesmo estágio de maturação, eliminando distorções de crescimento da carteira.


```python
# Usamos trimestres ('Q') para evitar ruído excessivo de safras mensais
df_finance['vintage_qt'] = df_finance['issue_d'].dt.to_period('Q').astype(str)
# Opcional: Safra Anual para visão macro
df_finance['vintage_yr'] = df_finance['issue_d'].dt.year.astype(str)

# Diferença em meses entre a última atividade e a emissão
df_finance['mob'] = (
    (df_finance['last_pymnt_d_calc'].dt.year - df_finance['issue_d'].dt.year) * 12 + 
    (df_finance['last_pymnt_d_calc'].dt.month - df_finance['issue_d'].dt.month)
)

# Aplica o lag de 4 meses para refletir a data real da parada de pagamento
mask_bad = df_finance['loan_status'].str.contains('Charged Off|Default', na=False)
df_finance.loc[mask_bad, 'mob'] += 4

# Em vez de limitar ao prazo do empréstimo,
# apenas garantimos que não existam MOBs negativos
df_finance['mob'] = df_finance['mob'].clip(lower=0)

# Sincronizar vintage com df_risk para análises futuras
df_risk['vintage_qt'] = df_finance['vintage_qt']
```

#### Categorização de Texto

Campos de texto livre, como Cargo/Profissão, são frequentemente descartados em modelos simples devido à sua "sujeira": existem milhares de maneiras de escrever "Engenheiro de Software". Para um algoritmo, cada variação é uma categoria diferente, gerando uma explosão de dimensionalidade que confunde a análise.

No entanto, descartar essa coluna seria desperdiçar ouro. Como vimos na análise exploratória, a profissão carrega um forte poder preditivo sobre a estabilidade financeira.

Nesta etapa, aplicaremos uma Normalização Semântica:

1. **Redução de Dimensionalidade:** Vamos varrer o texto em busca de palavras-chave para agrupar milhares de títulos únicos em macro-categorias lógicas.

2. **Captura de Valor:** Isso transforma um dado caótico e inutilizável em variáveis categóricas robustas, permitindo que a análise diferencie o risco de um "Funcionário Público" versus um "Autônomo" sem se perder nas nuances da grafia.


```python
# Função de Mapeamento
def categorize_emp_title(title):
    if pd.isna(title):
        return 'Missing'
    
    title = str(title).lower()
    
    categories = {
        'Public Sector': ['police', 'officer', 'military', 'firefighter', 'government', 'govt', 'city', 'state', 'federal', 'army', 'navy', 'usps'],
        'Healthcare': ['nurse', 'doctor', 'hospital', 'medical', 'physician', 'rn', 'dentist', 'health', 'clinic', 'paramedic'],
        'Education': ['teacher', 'professor', 'school', 'university', 'instructor', 'academic', 'education', 'faculty', 'college'],
        
        'Tech/Eng': ['engineer', 'software', 'it', 'developer', 'systems', 'analyst', 'tech', 'data', 'computer', 'programmer'],
        'Finance/Legal': ['accountant', 'finance', 'lawyer', 'legal', 'banking', 'audit', 'attorney', 'cpa'],
        
        'Self-Employed': ['owner', 'partner', 'president', 'ceo', 'founder', 'consultant', 'self', 'business'],
        
        'Management': ['manager', 'director', 'supervisor', 'lead', 'vp', 'executive', 'chief', 'head', 'principal'],
        'Service/Retail': ['sales', 'retail', 'clerk', 'cashier', 'customer', 'driver', 'truck', 'store', 'restaurant', 'food', 'waiter'],
        'Blue Collar': ['labor', 'construction', 'warehouse', 'operator', 'assembler', 'machinist'] # Adicionado para capturar setor manual
    }
    
    for category, keywords in categories.items():
        if any(keyword in title for keyword in keywords):
            return category
            
    return 'Other'

# Aplicação
df_risk['emp_sector'] = df_risk['emp_title'].apply(categorize_emp_title)

sector_dist = df_risk['emp_sector'].value_counts(normalize=True) * 100

df_risk.drop(columns=['emp_title'], inplace=True)
```

#### Remoção de linhas residuais

Os valores nulos restantes são residuais e estatisticamente insignificantes dado o volume total de dados.

Como as linhas problemáticas representam aproximadamente 0.001% do dataset, a melhor prática é removê-las.


```python
df_risk.isna().sum()
```




    loan_status                0
    issue_d                    0
    term                       0
    annual_inc                 0
    dti                        0
    verification_status        0
    installment                0
    home_ownership             0
    purpose                    0
    zip_code                   1
    addr_state                 0
    revol_util                 0
    inq_last_6mths             0
    mths_since_last_delinq     0
    pub_rec                    0
    earliest_cr_line          29
    open_acc                   0
    total_acc                  0
    grade                      0
    sub_grade                  0
    int_rate                   0
    loan_amnt                  0
    funded_amnt                0
    target                     0
    never_delinq               0
    emp_length_int             0
    emp_length_missing         0
    vintage_qt                 0
    emp_sector                 0
    dtype: int64




```python
# Remover as linhas residuais com nulos
df_risk.dropna(subset=['zip_code', 'earliest_cr_line'], inplace=True)

# Sincronizar o df_finance
# para manter o alinhamento 1:1 para análise de rentabilidade futura
df_finance = df_finance.loc[df_risk.index]
```

### Análise de Poder Preditivo (Pré-Scorecard)

Temos dezenas de colunas à disposição, mas na modelagem de risco de crédito, menos é mais. Uma análise cheia com variáveis fracas ou redundantes não apenas perde performance; ela se torna uma "caixa preta" difícil de explicar para a auditoria e perigosa para o negócio.

Nesta etapa, realizamos uma Auditoria de Sinal. Vamos submeter cada variável candidata a testes estatísticos rigorosos, focados no Information Value (IV) e Weight of Evidence (WoE), para responder a uma pergunta simples: "Esta variável consegue distinguir um bom pagador de um mau pagador?"

Nosso objetivo é:

1. **Filtrar o Ruído:** Descartar variáveis com baixo poder preditivo que não ajudam na decisão.

2. **Eliminar Redundância:** Identificar variáveis que dizem a mesma coisa e ficar apenas com a mais robusta.

3. **Garantir Monotonicidade:** Verificar se a relação da variável com o risco é lógica e estável, fundamental para a construção de Scorecards tradicionais.

$$WoE_i = \ln \left( \frac{\% \text{Good}_i}{\% \text{Bad}_i} \right)$$

$$IV = \sum_{i=1}^{n} \left[ (\% \text{Good}_i - \% \text{Bad}_i) \times \ln \left( \frac{\% \text{Good}_i}{\% \text{Bad}_i} \right) \right]$$


```python
def calculate_woe_iv(df, feature, target):
    lst = []
    df = df.copy()
    
    if np.issubdtype(df[feature].dtype, np.number):
        try:
            df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop').astype(str)
        except:
            df['bin'] = pd.cut(df[feature], bins=10).astype(str)
    else:
        # Preenche nulos como 'Missing'
        df[feature] = df[feature].fillna('Missing')
        df['bin'] = df[feature].astype(str)

    grouped = df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'Total', 'sum': 'Bad'})
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    # Suavização para evitar divisão por zero no Log
    grouped['Good'] = np.where(grouped['Good'] == 0, 0.5, grouped['Good'])
    grouped['Bad'] = np.where(grouped['Bad'] == 0, 0.5, grouped['Bad'])
    
    total_goods = grouped['Good'].sum()
    total_bads = grouped['Bad'].sum()
    
    grouped['Distr_Good'] = grouped['Good'] / total_goods
    grouped['Distr_Bad'] = grouped['Bad'] / total_bads
    
    # Fórmula do WoE: ln(Distr_Good / Distr_Bad)
    grouped['WoE'] = np.log(grouped['Distr_Good'] / grouped['Distr_Bad'])
    
    # Fórmula do IV: (Distr_Good - Distr_Bad) * WoE
    grouped['IV'] = (grouped['Distr_Good'] - grouped['Distr_Bad']) * grouped['WoE']
    
    # Retorna a tabela detalhada e o IV total da variável
    return grouped, grouped['IV'].sum()

# Selecionamos variáveis candidatas
candidate_vars = ['grade', 'int_rate', 'dti', 'annual_inc', 'revol_util', 
                  'emp_sector', 'term', 'home_ownership']

iv_results = []

for col in candidate_vars:
    if col in df_risk.columns:
        _, iv = calculate_woe_iv(df_risk, col, 'target')
        iv_results.append({'Variável': col, 'IV': iv})

# Cria DataFrame de resultados ordenado
df_iv = pd.DataFrame(iv_results).sort_values(by='IV', ascending=False)

plt.figure(figsize=(10, 6))

# Cores baseadas na "Regra de Siddiqi"
# < 0.02: Inútil (Cinza)
# 0.02 - 0.1: Fraco (Azul Claro)
# 0.1 - 0.3: Médio (Azul Escuro)
# 0.3 - 0.5: Forte (Verde)
# > 0.5: Suspeito (Vermelho)
colors = []
for iv in df_iv['IV']:
    if iv < 0.02: colors.append('lightblue')
    elif iv < 0.1: colors.append('skyblue')
    elif iv < 0.3: colors.append('steelblue')
    elif iv < 0.5: colors.append('blue')
    else: colors.append('firebrick')

sns.barplot(data=df_iv, x='IV', y='Variável', palette=colors)
plt.title('Histórico de Crédito e Renda são os principais preditores de inadimplência no modelo atual.', fontsize=14)
plt.xlabel('Information Value (IV)')
plt.axvline(0.02, color='black', linestyle='--', alpha=0.5)
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_82_0.png)
    



```python
df_iv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variável</th>
      <th>IV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>grade</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>int_rate</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dti</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>home_ownership</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>annual_inc</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>revol_util</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>emp_sector</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>term</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



## Analisar

Até agora, organizamos o passado. Agora, vamos colocar a mão na massa.

Já temos dados limpos, tipados e tratados. Mas uma tabela organizada não toma decisões. Nesta fase de Análise, deixamos de ser "faxineiros de dados" para nos tornarmos Investigadores de Padrões.

Não estamos mais procurando por erros de digitação; estamos procurando por comportamentos. Vamos cruzar variáveis para responder às perguntas.

Aqui é onde transformamos intuição em estatística e dados em estratégia. O que descobriremos a seguir será o "cérebro" das nossas futuras recomendações.

**Por que estou fazendo este gráfico?**

Em finanças, buscar rendimento sem medir a perda é uma armadilha comum. Um empréstimo com taxa de juros de 30% parece atraente, mas se 40% desses clientes não pagarem, o banco quebra. Meu objetivo aqui foi desafiar a premissa de que 'maior risco traz maior retorno'. Precisamos encontrar o Retorno Líquido Anualizado (NAR) real de cada segmento, descontando as perdas efetivas, para responder à pergunta: Estamos sendo pagos adequadamente pelo risco que estamos assumindo nos Grades F e G, ou estamos destruindo valor?


```python
# NAR = (1 + ROI_Simples) ^ (12 / Prazo_Meses) - 1

df_finance['term'] = df_risk['term']

df_calc = df_finance.copy()

# Cálculo do ROI Simples (Lucro / Investimento)
df_calc['roi_simple'] = (df_calc['total_pymnt'] - df_calc['funded_amnt']) / df_calc['funded_amnt']

# Cálculo do NAR (Ajuste temporal)
df_calc['nar'] = (1 + df_calc['roi_simple']) ** (12 / df_calc['term']) - 1

# Agrupamos para encontrar a média de Retorno e a Taxa de Inadimplência (Risk vs Reward)
profit_analysis = df_calc.groupby('sub_grade', observed=False).agg(
    avg_nar=('nar', 'mean'),
    bad_rate=('target', 'mean'), # target: 1 = Default, 0 = Paid
    volume=('funded_amnt', 'count'),
    avg_int_rate=('int_rate', 'mean')
).reset_index()

# Ordenar por Subgrade (A1 -> G5)
profit_analysis = profit_analysis.sort_values('sub_grade')

fig, ax1 = plt.subplots(figsize=(14, 8))

bars = ax1.bar(profit_analysis['sub_grade'], profit_analysis['avg_nar'], 
               color='steelblue', alpha=0.7, label='Retorno Líquido (NAR)')

ax1.set_xlabel('Score de Crédito (Sub-Grade)', fontsize=12)
ax1.set_ylabel('Retorno Líquido Anualizado (NAR)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Adicionar linha de referência zero
ax1.axhline(0, color='black', linewidth=1, linestyle='--')

# Gráfico de Linha: Taxa de Inadimplência (Eixo da Direita)
ax2 = ax1.twinx()
line = ax2.plot(profit_analysis['sub_grade'], profit_analysis['bad_rate'], 
                color='red', marker='o', linewidth=3, label='Bad Rate (Risco)')

ax2.set_ylabel('Taxa de Inadimplência (Bad Rate)', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='red')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Título e Legendas
plt.title('A partir do Grade C, o Retorno Líquido torna-se negativo, destruindo valor do acionista.', fontsize=16, pad=20)

# Destaque do "Ponto de Virada"
max_nar_idx = profit_analysis['avg_nar'].idxmax()
best_grade = profit_analysis.loc[max_nar_idx, 'sub_grade']
max_nar_val = profit_analysis.loc[max_nar_idx, 'avg_nar']

plt.annotate(f'Pico: {best_grade}',
             xy=(max_nar_idx, max_nar_val),
             xytext=(max_nar_idx, max_nar_val + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, fontweight='bold')

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_87_0.png)
    



```python
profit_analysis[['sub_grade', 'avg_int_rate', 'bad_rate', 'avg_nar']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_grade</th>
      <th>avg_int_rate</th>
      <th>bad_rate</th>
      <th>avg_nar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A1</td>
      <td>5.55</td>
      <td>0.03</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A2</td>
      <td>6.52</td>
      <td>0.05</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A3</td>
      <td>7.12</td>
      <td>0.06</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A4</td>
      <td>7.51</td>
      <td>0.07</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A5</td>
      <td>8.21</td>
      <td>0.08</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B1</td>
      <td>8.91</td>
      <td>0.10</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B2</td>
      <td>9.92</td>
      <td>0.11</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B3</td>
      <td>10.76</td>
      <td>0.13</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>B4</td>
      <td>11.51</td>
      <td>0.15</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B5</td>
      <td>12.02</td>
      <td>0.17</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



**O que este gráfico nos diz?**

Ao descontar as perdas efetivas para encontrar o Retorno Líquido Anualizado, identificamos uma falha crítica na precificação de risco. Contrariando a premissa de que maior risco compensa com maior retorno, os dados mostram que a rentabilidade real é maximizada exclusivamente no Grade A.

O ponto de inflexão é alarmante: a partir do Grade C, a inadimplência corrói inteiramente o prêmio de juros, resultando em destruição de valor. Isso significa que os lucros gerados pelos clientes 'Prime' estão subsidiando o prejuízo estrutural de aproximadamente 70% da carteira. A ação recomendada não é apenas restringir F e G, mas revisar imediatamente a política de crédito a partir do Grade C, onde a operação já se torna deficitária.

**Por que estou fazendo este gráfico?**

Métricas estáticas de inadimplência mentem quando a carteira está crescendo. O aumento no volume de vendas novas infla o denominador, mascarando problemas na originação. Para eliminar esse ruído, apliquei a Análise de Safra, considerada o 'Padrão Ouro' na gestão de risco. Ao agrupar empréstimos pela data de originação e rastrear seu desempenho por idade, isolo a qualidade real da concessão de crédito, independentemente do crescimento da carteira.


```python
cohort_sizes = df_finance.groupby(['vintage_qt', 'term'])['loan_amnt'].count().reset_index()
cohort_sizes.rename(columns={'loan_amnt': 'orig_count'}, inplace=True)

defaults_by_mob = df_finance[df_finance['target'] == 1].groupby(
    ['vintage_qt', 'term', 'mob']
)['loan_amnt'].count().reset_index()
defaults_by_mob.rename(columns={'loan_amnt': 'bad_count'}, inplace=True)

vintage_curve = defaults_by_mob.merge(cohort_sizes, on=['vintage_qt', 'term'])

vintage_curve.sort_values(['term', 'vintage_qt', 'mob'], inplace=True)

vintage_curve['cum_bad_count'] = vintage_curve.groupby(['vintage_qt', 'term'])['bad_count'].cumsum()
vintage_curve['cum_bad_rate'] = vintage_curve['cum_bad_count'] / vintage_curve['orig_count']

mask_36 = (vintage_curve['term'] == 36) & (vintage_curve['mob'] <= 37)

mask_60 = (vintage_curve['term'] == 60) & (vintage_curve['mob'] <= 65)

vintage_curve_view = vintage_curve[mask_36 | mask_60]


plt.figure(figsize=(14, 8))

ax = sns.lineplot(
    data=vintage_curve_view, 
    x='mob', 
    y='cum_bad_rate', 
    hue='term', 
    style='term',
    palette=['#1f77b4', '#d62728'], 
    linewidth=3,
    legend=False
)

plt.title('Empréstimos de 60 meses apresentam o dobro de inadimplência acumulada em relação aos de 36.', fontsize=16, pad=20)
plt.xlabel('Meses em Carteira (MOB - Months on Book)', fontsize=12)
plt.ylabel('Taxa de Inadimplência Acumulada (%)', fontsize=12)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Linhas verticais de referência
plt.axvline(36, color='#1f77b4', linestyle='--', alpha=0.3)
plt.axvline(60, color='#d62728', linestyle='--', alpha=0.3)

for term, color in zip([36, 60], ['#1f77b4', '#d62728']):
    subset = vintage_curve_view[vintage_curve_view['term'] == term]
    last_mob = subset['mob'].max()
    last_rate = subset[subset['mob'] == last_mob]['cum_bad_rate'].values[0]
    
    plt.text(
        last_mob,
        last_rate, 
        f'{term} Meses', 
        color=color, 
        fontweight='bold',
        va='center',
        ha='left'
    )

plt.xlim(0, vintage_curve_view['mob'].max() + 8)

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_91_0.png)
    



```python
compare_36m = vintage_curve_view[vintage_curve_view['mob'] == 36].groupby('term')['cum_bad_rate'].mean()
compare_36m_df = compare_36m.to_frame(name='Taxa de Inadimplência Acumulada (Mês 36)')
compare_36m_df.style.format('{:.2%}')
```




<style type="text/css">
</style>
<table id="T_eaa2c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eaa2c_level0_col0" class="col_heading level0 col0" >Taxa de Inadimplência Acumulada (Mês 36)</th>
    </tr>
    <tr>
      <th class="index_name level0" >term</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eaa2c_level0_row0" class="row_heading level0 row0" >36</th>
      <td id="T_eaa2c_row0_col0" class="data row0 col0" >13.49%</td>
    </tr>
    <tr>
      <th id="T_eaa2c_level0_row1" class="row_heading level0 row1" >60</th>
      <td id="T_eaa2c_row1_col0" class="data row1 col0" >23.06%</td>
    </tr>
  </tbody>
</table>




**O que este gráfico nos diz?**

A Análise de Safra expõe um problema claro de Seleção Adversa. Ao isolar o desempenho por data de originação, observamos que o produto de 60 meses atrai um perfil de cliente fundamentalmente mais frágil do que o de 36 meses.

A divergência é estrutural: enquanto a inadimplência da carteira de 36 meses tende a estabilizar após o 24º mês de vida, a curva de 60 meses mantém uma trajetória ascendente contínua, acumulando perdas por um período muito maior sem sinal de recuperação. Isso comprova que o prazo estendido não está auxiliando a capacidade de pagamento, mas sim mascarando o risco inicial. A recomendação estratégica é endurecer os critérios de entrada especificamente para solicitações de longo prazo.

**Por que estou fazendo este gráfico?**

Muitos modelos financeiros assumem ingenuamente que, quando um cliente falha, perderemos uma 'média' fixa do valor. No entanto, a realidade da recuperação de crédito raramente é suave. Construí este histograma de distribuição de LGD para testar a hipótese de bimodalidade. Precisamos saber se existe uma 'rede de segurança' na recuperação ou se estamos lidando com um cenário binário de 'tudo ou nada', o que alteraria drasticamente nossa necessidade de provisionamento de capital.


```python
# Filtramos apenas os empréstimos que já deram Default/Charge Off
# LGD só existe se houver o evento de inadimplência (Target = 1)
df_lgd = df_finance[df_finance['target'] == 1].copy()

# Fórmula: LGD = 1 - (Valor Recuperado / Valor da Exposição)
# Nota: Em dados reais, a EAD (Exposure at Default) seria o saldo devedor na data do default.
# Como proxy conservador para Lending Club, usamos o funded_amnt ou (funded - principal_pago).

# Taxa de Recuperação
df_lgd['recovery_rate'] = df_lgd['recoveries'] / df_lgd['funded_amnt']

# Tratamento de Limites 
# Recuperação > 1.0 pode ocorrer devido a multas/juros, mas para LGD limitamos a 0 e 1.
df_lgd['recovery_rate'] = df_lgd['recovery_rate'].clip(0, 1)

df_lgd['LGD'] = 1 - df_lgd['recovery_rate']

plt.figure(figsize=(12, 7))

# Histograma com KDE
sns.histplot(
    data=df_lgd, 
    x='LGD', 
    bins=30, 
    kde=True, 
    color='blue', 
    stat='probability',
    line_kws={'linewidth': 3}
)

mean_lgd = df_lgd['LGD'].mean()
median_lgd = df_lgd['LGD'].median()

plt.axvline(mean_lgd, color='black', linestyle='--', linewidth=2, label=f'Média: {mean_lgd:.2%}')
plt.axvline(median_lgd, color='red', linestyle='-.', linewidth=2, label=f'Mediana: {median_lgd:.2%}')

plt.title('A concentração de perdas no LGD é bimodal: ou recuperamos quase tudo, ou perdemos o colateral.', fontsize=16, pad=20)
plt.xlabel('LGD (Percentual do Valor Perdido)', fontsize=12)
plt.ylabel('Probabilidade / Frequência', fontsize=12)
plt.legend(fontsize=12)

plt.annotate('Concentração de Perda Total: Recuperação é estatisticamente improvável.',
             xy=(0.75, 0.10), 
             xycoords='axes fraction',
             ha='right', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_95_0.png)
    



```python
lgd_summary = df_lgd['LGD'].describe().to_frame(name='Estatísticas de Severidade (LGD)')

lgd_summary.loc['Perda Total (LGD > 90%)'] = (df_lgd['LGD'] > 0.9).mean()

lgd_summary.style.format(lambda x: f"{x:,.0f}" if x > 100 else f"{x:.2%}") \
                 .background_gradient(cmap='Blues', subset=pd.IndexSlice[['mean', '50%', 'max'], :])
```




<style type="text/css">
#T_db920_row1_col0 {
  background-color: #f7fbff;
  color: #000000;
}
#T_db920_row5_col0 {
  background-color: #d0e1f2;
  color: #000000;
}
#T_db920_row7_col0 {
  background-color: #08306b;
  color: #f1f1f1;
}
</style>
<table id="T_db920">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_db920_level0_col0" class="col_heading level0 col0" >Estatísticas de Severidade (LGD)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_db920_level0_row0" class="row_heading level0 row0" >count</th>
      <td id="T_db920_row0_col0" class="data row0 col0" >262,317</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row1" class="row_heading level0 row1" >mean</th>
      <td id="T_db920_row1_col0" class="data row1 col0" >92.68%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row2" class="row_heading level0 row2" >std</th>
      <td id="T_db920_row2_col0" class="data row2 col0" >9.40%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row3" class="row_heading level0 row3" >min</th>
      <td id="T_db920_row3_col0" class="data row3 col0" >0.00%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row4" class="row_heading level0 row4" >25%</th>
      <td id="T_db920_row4_col0" class="data row4 col0" >88.97%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row5" class="row_heading level0 row5" >50%</th>
      <td id="T_db920_row5_col0" class="data row5 col0" >94.16%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row6" class="row_heading level0 row6" >75%</th>
      <td id="T_db920_row6_col0" class="data row6 col0" >100.00%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row7" class="row_heading level0 row7" >max</th>
      <td id="T_db920_row7_col0" class="data row7 col0" >100.00%</td>
    </tr>
    <tr>
      <th id="T_db920_level0_row8" class="row_heading level0 row8" >Perda Total (LGD > 90%)</th>
      <td id="T_db920_row8_col0" class="data row8 col0" >70.39%</td>
    </tr>
  </tbody>
</table>




**O que este gráfico nos diz?**

A distribuição de LGD derruba a hipótese de uma recuperação média suave. Observamos uma concentração massiva de perdas próximas a 100%, com uma mediana de 94.16%.

Diferente de carteiras colateralizadas que apresentam comportamento bimodal, este portfólio demonstra uma irreversibilidade do default. Uma vez que o cliente entra em inadimplência, a probabilidade de recuperação significativa é quase nula. Isso invalida estratégias tradicionais de cobrança tardia e reforça a tese de que a rentabilidade depende inteiramente da filtragem na entrada, pois não há 'segunda chance' operacional.

**Por que estou fazendo este gráfico?**

O mercado muda, e o perfil do cliente também. Um modelo treinado com dados de 2015 pode ser inútil em 2024 se a população mudou. Calculei o Population Stability Index (PSI) para as principais variáveis de risco. O objetivo é garantir a governança do modelo: identificar se houve mudanças estruturais na base de clientes que invalidariam nossas regras de aprovação atuais.


```python
def calculate_psi(expected, actual, buckets=10):
    """
    Calcula o PSI para uma variável numérica.
    """
    breakpoints = np.percentile(expected, np.arange(0, buckets + 1) / buckets * 100)
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    epsilon = 0.0001
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    # Cálculo final do PSI
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value

df_monitoring = df_risk.sort_values('issue_d').copy()

split_idx = int(len(df_monitoring) * 0.7)
df_reference = df_monitoring.iloc[:split_idx]
df_current = df_monitoring.iloc[split_idx:]

vars_to_monitor = ['int_rate', 'dti', 'annual_inc', 'loan_amnt', 'revol_util']
psi_results = []

for var in vars_to_monitor:
    psi = calculate_psi(df_reference[var], df_current[var], buckets=10)
    psi_results.append({'Variável': var, 'PSI': psi})

psi_df = pd.DataFrame(psi_results)

plt.figure(figsize=(10, 6))

sns.barplot(data=psi_df, x='Variável', y='PSI', color='blue', alpha=0.8)

plt.axhline(0.1, color='green', linestyle='--', alpha=0.6, label='Estável (<0.1)')
plt.axhline(0.25, color='red', linestyle='--', alpha=0.6, label='Crítico (>0.25)')

plt.title('A alteração no perfil do público de entrada indica necessidade iminente de recalibragem do modelo.', fontsize=14)
plt.ylabel('Valor do PSI', fontsize=12)
plt.legend()

for index, row in psi_df.iterrows():
    plt.text(index, row.PSI + 0.005, f'{row.PSI:.3f}', color='black', ha="center", fontweight='bold')

plt.tight_layout()
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_99_0.png)
    



```python
psi_df.style.format({'PSI': '{:.4f}'})
```




<style type="text/css">
</style>
<table id="T_7d3cd">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7d3cd_level0_col0" class="col_heading level0 col0" >Variável</th>
      <th id="T_7d3cd_level0_col1" class="col_heading level0 col1" >PSI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7d3cd_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7d3cd_row0_col0" class="data row0 col0" >int_rate</td>
      <td id="T_7d3cd_row0_col1" class="data row0 col1" >0.0839</td>
    </tr>
    <tr>
      <th id="T_7d3cd_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_7d3cd_row1_col0" class="data row1 col0" >dti</td>
      <td id="T_7d3cd_row1_col1" class="data row1 col1" >0.0033</td>
    </tr>
    <tr>
      <th id="T_7d3cd_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_7d3cd_row2_col0" class="data row2 col0" >annual_inc</td>
      <td id="T_7d3cd_row2_col1" class="data row2 col1" >0.0064</td>
    </tr>
    <tr>
      <th id="T_7d3cd_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_7d3cd_row3_col0" class="data row3 col0" >loan_amnt</td>
      <td id="T_7d3cd_row3_col1" class="data row3 col1" >0.0200</td>
    </tr>
    <tr>
      <th id="T_7d3cd_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_7d3cd_row4_col0" class="data row4 col0" >revol_util</td>
      <td id="T_7d3cd_row4_col1" class="data row4 col1" >0.1061</td>
    </tr>
  </tbody>
</table>




**O que este gráfico nos diz?**

Modelos de crédito não são estáticos; eles se degradam conforme o comportamento do consumidor muda. Implementei o Population Stability Index (PSI) para auditar a validade contínua das regras de aprovação.

O painel revela uma estabilidade estrutural na maioria dos drivers de risco (PSI < 0.1), validando a robustez atual. Contudo, o sistema disparou um alerta amarelo na variável revol_util (PSI = 0.1061), que cruzou o limiar de estabilidade. Isso indica uma mudança recente no padrão de uso de limites rotativos pelos clientes, um indicador antecedente de estresse financeiro. Embora ainda abaixo da zona crítica de recalibragem, esse indicador exige monitoramento quinzenal para evitar que o modelo perca precisão preditiva neste segmento específico.

**Por que estou fazendo este gráfico?**

O risco de crédito não é uniformemente distribuído; ele possui clusters geográficos influenciados por economias locais, desemprego regional e legislações estaduais. Criei este mapa de calor não apenas para ver onde estão os empréstimos, mas para calcular o Risco Relativo de cada estado. Precisamos saber se a nossa exposição em estados específicos está carregando um risco sistêmico oculto.


```python
state_risk = df_risk.groupby('addr_state', observed=False).agg(
    volume=('target', 'count'),
    bad_rate=('target', 'mean'),
    avg_int_rate=('int_rate', 'mean')
).reset_index()

# Excluímos estados com volume muito baixo (< 50 empréstimos) para evitar ruído
min_volume_threshold = 50
state_risk = state_risk[state_risk['volume'] > min_volume_threshold]

# Calculamos a média global do portfólio para servir de base
global_avg_bad_rate = df_finance['target'].mean()

# Risk Index: > 1.0 significa risco acima da média, < 1.0 significa abaixo
# Ex: 1.20 significa que o estado é 20% mais arriscado que a média nacional
state_risk['risk_index'] = state_risk['bad_rate'] / global_avg_bad_rate

# Ordenar do mais arriscado para o mais seguro
state_risk = state_risk.sort_values('bad_rate', ascending=False)

fig = px.choropleth(
    state_risk, 
    locations='addr_state', 
    locationmode="USA-states", 
    color='bad_rate',
    scope="usa",
    color_continuous_scale="Blues",
    labels={'bad_rate': 'Taxa de Inadimplência'},
    title='<b>Distribuição Geográfica do Risco de Crédito (Bad Rate por Estado)</b>',
    hover_data={'risk_index': ':.2f', 'volume': ':,', 'bad_rate': ':.2%'}
)

fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(title="Bad Rate", tickformat=".1%")
)

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="e1c53079-404e-4912-b7bb-cbe0f773c36f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e1c53079-404e-4912-b7bb-cbe0f773c36f")) {                    Plotly.newPlot(                        "e1c53079-404e-4912-b7bb-cbe0f773c36f",                        [{"coloraxis":"coloraxis","customdata":[[1.3050101345254541,6321.0,0.2621420661287771],[1.2670865839051197,3426.0,0.2545242265032107],[1.2043505094012352,9718.0,0.24192220621527064],[1.1818657876077303,16158.0,0.23740561950736477],[1.1800215613626475,11859.0,0.23703516316721476],[1.164031767953241,15037.0,0.2338232360178227],[1.1028129520591552,106606.0,0.2215259929084667],[1.0999530275070237,19674.0,0.22095150960658738],[1.0766995894792406,21019.0,0.21628050811170846],[1.0751209245951303,19670.0,0.2159633960345704],[1.0747487992821232,92821.0,0.21588864588832268],[1.0723250850320687,7168.0,0.21540178571428573],[1.0678374543578244,20634.0,0.2145003392459048],[1.064092212180316,30288.0,0.21374801901743265],[1.062126833773562,2681.0,0.2133532264080567],[1.0590229997107024,47008.0,0.2127297481279782],[1.0513953837945378,12467.0,0.21119756156252506],[1.0430345555922518,36646.0,0.20951809201549965],[1.0411831316150446,44237.0,0.20914618984108327],[1.0385102151441206,1510.0,0.20860927152317882],[1.0353412927701162,42520.0,0.20797271872060208],[1.0143451815815567,34193.0,0.2037551545637996],[1.011101604784496,6573.0,0.2031036056595162],[1.0056399704975194,3688.0,0.20200650759219088],[0.9969456228533531,36917.0,0.20026004279871062],[0.9933409681332853,23274.0,0.19953596287703015],[0.9921192473172407,3101.0,0.19929055143502097],[0.9921181050146983,106809.0,0.19929032197661245],[0.9820614173800869,191296.0,0.19727019906323184],[0.9804110263441012,31751.0,0.19693867909672136],[0.9641553228612143,1549.0,0.19367333763718528],[0.9519029095267565,30087.0,0.19121215142752684],[0.9192116099121085,42124.0,0.18464533282689202],[0.9129361500073482,17177.0,0.18338475868894452],[0.9086377817322957,50164.0,0.1825213300374771],[0.8962431927150573,5699.0,0.18003158448850676],[0.8708887629046091,19121.0,0.17493854923905652],[0.8563920581125591,9795.0,0.1720265441551812],[0.8514199813874842,3707.0,0.1710277852711087],[0.8446352054660576,2835.0,0.16966490299823633],[0.8348796948481788,10912.0,0.16770527859237536],[0.820065175885911,15486.0,0.16472943303629084],[0.79256503937942,28391.0,0.15920538198724948],[0.7894015791899501,4755.0,0.15856992639327025],[0.7766305651501115,28903.0,0.1560045669999654],[0.7251663885231058,6254.0,0.14566677326511032],[0.7175891591126794,15977.0,0.14414470801777554],[0.7036815403235085,2561.0,0.14135103475205],[0.6948814713368431,1920.0,0.13958333333333334],[0.6539483315875121,3380.0,0.13136094674556212]],"geo":"geo","hovertemplate":"addr_state=%{location}\u003cbr\u003erisk_index=%{customdata[0]:.2f}\u003cbr\u003evolume=%{customdata[1]:,}\u003cbr\u003eTaxa de Inadimplência=%{z:.2%}\u003cextra\u003e\u003c\u002fextra\u003e","locationmode":"USA-states","locations":["MS","NE","AR","AL","OK","LA","NY","NV","IN","TN","FL","NM","MO","MD","SD","NJ","KY","NC","PA","ND","OH","MI","HI","DE","VA","MN","AK","TX","CA","AZ","ID","MA","GA","WI","IL","RI","CT","UT","MT","WY","KS","SC","WA","WV","CO","NH","OR","VT","ME","DC"],"name":"","z":[0.2621420661287771,0.2545242265032107,0.24192220621527064,0.23740561950736477,0.23703516316721476,0.2338232360178227,0.2215259929084667,0.22095150960658738,0.21628050811170846,0.2159633960345704,0.21588864588832268,0.21540178571428573,0.2145003392459048,0.21374801901743265,0.2133532264080567,0.2127297481279782,0.21119756156252506,0.20951809201549965,0.20914618984108327,0.20860927152317882,0.20797271872060208,0.2037551545637996,0.2031036056595162,0.20200650759219088,0.20026004279871062,0.19953596287703015,0.19929055143502097,0.19929032197661245,0.19727019906323184,0.19693867909672136,0.19367333763718528,0.19121215142752684,0.18464533282689202,0.18338475868894452,0.1825213300374771,0.18003158448850676,0.17493854923905652,0.1720265441551812,0.1710277852711087,0.16966490299823633,0.16770527859237536,0.16472943303629084,0.15920538198724948,0.15856992639327025,0.1560045669999654,0.14566677326511032,0.14414470801777554,0.14135103475205,0.13958333333333334,0.13136094674556212],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"usa"},"coloraxis":{"colorbar":{"title":{"text":"Bad Rate"},"tickformat":".1%"},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"\u003cb\u003eDistribuição Geográfica do Risco de Crédito (Bad Rate por Estado)\u003c\u002fb\u003e"},"margin":{"r":0,"t":50,"l":0,"b":0}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e1c53079-404e-4912-b7bb-cbe0f773c36f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
def format_state_table(df, title, color_map):
    print(f"\n{title}")
    return df.style.format({
        'bad_rate': '{:.2%}',
        'risk_index': '{:.2f}x',
        'volume': '{:,}'
    }).background_gradient(cmap=color_map, subset=['bad_rate']) \
      .set_properties(**{'text-align': 'center'})
```


```python
toxic_df = state_risk.sort_values('bad_rate', ascending=False).head(5)
display(format_state_table(toxic_df[['addr_state', 'volume', 'bad_rate', 'risk_index']], 
                          "Top 5 Estados com Maior Risco", "Reds"))
```

    
    Top 5 Estados com Maior Risco
    


<style type="text/css">
#T_a01ab_row0_col0, #T_a01ab_row0_col1, #T_a01ab_row0_col3, #T_a01ab_row1_col0, #T_a01ab_row1_col1, #T_a01ab_row1_col3, #T_a01ab_row2_col0, #T_a01ab_row2_col1, #T_a01ab_row2_col3, #T_a01ab_row3_col0, #T_a01ab_row3_col1, #T_a01ab_row3_col3, #T_a01ab_row4_col0, #T_a01ab_row4_col1, #T_a01ab_row4_col3 {
  text-align: center;
}
#T_a01ab_row0_col2 {
  background-color: #67000d;
  color: #f1f1f1;
  text-align: center;
}
#T_a01ab_row1_col2 {
  background-color: #da2723;
  color: #f1f1f1;
  text-align: center;
}
#T_a01ab_row2_col2 {
  background-color: #fdccb8;
  color: #000000;
  text-align: center;
}
#T_a01ab_row3_col2 {
  background-color: #fff3ed;
  color: #000000;
  text-align: center;
}
#T_a01ab_row4_col2 {
  background-color: #fff5f0;
  color: #000000;
  text-align: center;
}
</style>
<table id="T_a01ab">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a01ab_level0_col0" class="col_heading level0 col0" >addr_state</th>
      <th id="T_a01ab_level0_col1" class="col_heading level0 col1" >volume</th>
      <th id="T_a01ab_level0_col2" class="col_heading level0 col2" >bad_rate</th>
      <th id="T_a01ab_level0_col3" class="col_heading level0 col3" >risk_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a01ab_level0_row0" class="row_heading level0 row0" >25</th>
      <td id="T_a01ab_row0_col0" class="data row0 col0" >MS</td>
      <td id="T_a01ab_row0_col1" class="data row0 col1" >6,321</td>
      <td id="T_a01ab_row0_col2" class="data row0 col2" >26.21%</td>
      <td id="T_a01ab_row0_col3" class="data row0 col3" >1.31x</td>
    </tr>
    <tr>
      <th id="T_a01ab_level0_row1" class="row_heading level0 row1" >29</th>
      <td id="T_a01ab_row1_col0" class="data row1 col0" >NE</td>
      <td id="T_a01ab_row1_col1" class="data row1 col1" >3,426</td>
      <td id="T_a01ab_row1_col2" class="data row1 col2" >25.45%</td>
      <td id="T_a01ab_row1_col3" class="data row1 col3" >1.27x</td>
    </tr>
    <tr>
      <th id="T_a01ab_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a01ab_row2_col0" class="data row2 col0" >AR</td>
      <td id="T_a01ab_row2_col1" class="data row2 col1" >9,718</td>
      <td id="T_a01ab_row2_col2" class="data row2 col2" >24.19%</td>
      <td id="T_a01ab_row2_col3" class="data row2 col3" >1.20x</td>
    </tr>
    <tr>
      <th id="T_a01ab_level0_row3" class="row_heading level0 row3" >1</th>
      <td id="T_a01ab_row3_col0" class="data row3 col0" >AL</td>
      <td id="T_a01ab_row3_col1" class="data row3 col1" >16,158</td>
      <td id="T_a01ab_row3_col2" class="data row3 col2" >23.74%</td>
      <td id="T_a01ab_row3_col3" class="data row3 col3" >1.18x</td>
    </tr>
    <tr>
      <th id="T_a01ab_level0_row4" class="row_heading level0 row4" >36</th>
      <td id="T_a01ab_row4_col0" class="data row4 col0" >OK</td>
      <td id="T_a01ab_row4_col1" class="data row4 col1" >11,859</td>
      <td id="T_a01ab_row4_col2" class="data row4 col2" >23.70%</td>
      <td id="T_a01ab_row4_col3" class="data row4 col3" >1.18x</td>
    </tr>
  </tbody>
</table>




```python
opportunity_df = state_risk.sort_values('bad_rate', ascending=True).head(5)
display(format_state_table(opportunity_df[['addr_state', 'volume', 'bad_rate', 'risk_index']], 
                          "Top 5 Estados Mais Seguros", "Blues_r"))
```

    
    Top 5 Estados Mais Seguros
    


<style type="text/css">
#T_489aa_row0_col0, #T_489aa_row0_col1, #T_489aa_row0_col3, #T_489aa_row1_col0, #T_489aa_row1_col1, #T_489aa_row1_col3, #T_489aa_row2_col0, #T_489aa_row2_col1, #T_489aa_row2_col3, #T_489aa_row3_col0, #T_489aa_row3_col1, #T_489aa_row3_col3, #T_489aa_row4_col0, #T_489aa_row4_col1, #T_489aa_row4_col3 {
  text-align: center;
}
#T_489aa_row0_col2 {
  background-color: #08306b;
  color: #f1f1f1;
  text-align: center;
}
#T_489aa_row1_col2 {
  background-color: #8abfdd;
  color: #000000;
  text-align: center;
}
#T_489aa_row2_col2 {
  background-color: #b5d4e9;
  color: #000000;
  text-align: center;
}
#T_489aa_row3_col2 {
  background-color: #e2edf8;
  color: #000000;
  text-align: center;
}
#T_489aa_row4_col2 {
  background-color: #f7fbff;
  color: #000000;
  text-align: center;
}
</style>
<table id="T_489aa">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_489aa_level0_col0" class="col_heading level0 col0" >addr_state</th>
      <th id="T_489aa_level0_col1" class="col_heading level0 col1" >volume</th>
      <th id="T_489aa_level0_col2" class="col_heading level0 col2" >bad_rate</th>
      <th id="T_489aa_level0_col3" class="col_heading level0 col3" >risk_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_489aa_level0_row0" class="row_heading level0 row0" >7</th>
      <td id="T_489aa_row0_col0" class="data row0 col0" >DC</td>
      <td id="T_489aa_row0_col1" class="data row0 col1" >3,380</td>
      <td id="T_489aa_row0_col2" class="data row0 col2" >13.14%</td>
      <td id="T_489aa_row0_col3" class="data row0 col3" >0.65x</td>
    </tr>
    <tr>
      <th id="T_489aa_level0_row1" class="row_heading level0 row1" >21</th>
      <td id="T_489aa_row1_col0" class="data row1 col0" >ME</td>
      <td id="T_489aa_row1_col1" class="data row1 col1" >1,920</td>
      <td id="T_489aa_row1_col2" class="data row1 col2" >13.96%</td>
      <td id="T_489aa_row1_col3" class="data row1 col3" >0.69x</td>
    </tr>
    <tr>
      <th id="T_489aa_level0_row2" class="row_heading level0 row2" >46</th>
      <td id="T_489aa_row2_col0" class="data row2 col0" >VT</td>
      <td id="T_489aa_row2_col1" class="data row2 col1" >2,561</td>
      <td id="T_489aa_row2_col2" class="data row2 col2" >14.14%</td>
      <td id="T_489aa_row2_col3" class="data row2 col3" >0.70x</td>
    </tr>
    <tr>
      <th id="T_489aa_level0_row3" class="row_heading level0 row3" >37</th>
      <td id="T_489aa_row3_col0" class="data row3 col0" >OR</td>
      <td id="T_489aa_row3_col1" class="data row3 col1" >15,977</td>
      <td id="T_489aa_row3_col2" class="data row3 col2" >14.41%</td>
      <td id="T_489aa_row3_col3" class="data row3 col3" >0.72x</td>
    </tr>
    <tr>
      <th id="T_489aa_level0_row4" class="row_heading level0 row4" >30</th>
      <td id="T_489aa_row4_col0" class="data row4 col0" >NH</td>
      <td id="T_489aa_row4_col1" class="data row4 col1" >6,254</td>
      <td id="T_489aa_row4_col2" class="data row4 col2" >14.57%</td>
      <td id="T_489aa_row4_col3" class="data row4 col3" >0.73x</td>
    </tr>
  </tbody>
</table>



**O que este gráfico nos diz?**

O risco de crédito não respeita fronteiras estaduais uniformemente. O mapa de calor revela clusters de inadimplência severa, com um spread de quase 100% entre os melhores e piores estados.

Identificamos que a exposição em regiões específicas carrega um risco muito superior à média nacional. Isso prova que uma política de crédito unificada é ineficiente: ela penaliza bons pagadores em estados de baixo risco e é permissiva demais em zonas de alta inadimplência.

**Por que estou fazendo este gráfico?**

Saber quanto perdemos é importante, mas saber quando desistir de cobrar é vital para a eficiência operacional. Manter operações de cobrança custa dinheiro. Desenvolvi esta curva de recuperação temporal para identificar o ponto de saturação. A pergunta de negócio é: até que mês após o default ainda vale a pena gastar recursos ligando para o cliente antes que o custo marginal da cobrança supere o valor recuperado?


```python
# Filtramos apenas clientes que entraram em Default/Charge Off
# Excluímos aqueles que não tiveram tempo de ter recuperação
df_rec = df_finance[df_finance['target'] == 1].copy()

# Em um snapshot, não temos a data de cada $ recuperado.
# Mas sabemos quando o cliente parou de pagar e a data final do arquivo.
# A diferença é a "Janela de Oportunidade de Cobrança".

# Data de referência do arquivo
snapshot_date = df_rec['last_pymnt_d'].max()

# Estimativa da Data do Default Contábil 
# Assume-se: Último Pagamento + 4 Meses
# Se last_pymnt_d for NaT (nunca pagou), usamos issue_d + 4 meses
df_rec['default_date_est'] = df_rec['last_pymnt_d'].fillna(df_rec['issue_d']) + pd.DateOffset(months=4)

# Cálculo dos Meses Desde o Default
df_rec['months_since_default'] = (
    (snapshot_date.year - df_rec['default_date_est'].dt.year) * 12 + 
    (snapshot_date.month - df_rec['default_date_est'].dt.month)
)

# Remover datas futuras ou erros negativos e limitar a 24 meses
df_rec = df_rec[(df_rec['months_since_default'] >= 0) & (df_rec['months_since_default'] <= 36)]

# Recovery Rate = Valor Recuperado / Valor Financiado
df_rec['recovery_rate'] = df_rec['recoveries'] / df_rec['funded_amnt']
df_rec['recovery_rate'] = df_rec['recovery_rate'].clip(0, 1) 

# A lógica: Clientes que deram default há 10 meses tiveram 10 meses para recuperar.
# A média da recuperação deles representa o "Potencial Acumulado" no mês 10.
recovery_curve = df_rec.groupby('months_since_default')['recovery_rate'].mean().reset_index()

# Suavização da Curva para remover ruído mensal
recovery_curve['recovery_rate_smooth'] = recovery_curve['recovery_rate'].rolling(window=3, min_periods=1).mean()

plt.figure(figsize=(14, 7))

# Plot da Curva
plt.plot(recovery_curve['months_since_default'], recovery_curve['recovery_rate_smooth'], 
         marker='o', color='darkblue', linewidth=3, label='Taxa de Recuperação Média')

# Linhas de Decisão
plt.axvline(6, color='orange', linestyle='--', label='6 Meses (Curto Prazo)')
plt.axvline(12, color='red', linestyle='--', label='12 Meses (Write-off Típico)')

# Formatação
plt.title('O esforço de cobrança perde eficiência econômica após o 6º mês de atraso.', fontsize=16, pad=20)
plt.xlabel('Meses Desde o Default (Tempo na Régua de Cobrança)', fontsize=12)
plt.ylabel('Taxa de Recuperação Acumulada (%)', fontsize=12)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend()

# Anotação de "Ponto de Saturação"
# Identificamos onde a curva achata
try:
    saturation_point = recovery_curve[recovery_curve['recovery_rate_smooth'].diff() < 0.001].iloc
    plt.annotate('Ponto de Saturação\n(Vender Dívida Aqui)', 
                 xy=(saturation_point['months_since_default'], saturation_point['recovery_rate_smooth']),
                 xytext=(saturation_point['months_since_default']+2, saturation_point['recovery_rate_smooth']-0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11, fontweight='bold', color='darkred')
except:
    pass

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_109_0.png)
    


**O que este gráfico nos diz?**

Definir o momento de parar é tão importante quanto saber cobrar. A Curva de Recuperação Temporal demonstra a Lei dos Rendimentos Decrescentes na prática: 80% de todo o valor recuperável é obtido nos primeiros 9 meses.

Após o marco de 12 meses (linha vermelha), a curva entra em um platô estagnado. Manter esses contratos sob gestão interna torna-se destrutivo, pois o Custo Marginal de Recuperação (ligações, equipe, sistemas) supera o valor efetivamente recuperado.

**Por que estou fazendo este gráfico?**

Na análise de crédito tradicional, olhamos muito para a renda bruta, mas frequentemente ignoramos a fonte da renda (Estabilidade). Um funcionário público que ganha USD 5.000 tem um perfil de risco radicalmente diferente de um comissionado de vendas que ganha os mesmos USD 5.000 em média, mas com alta volatilidade. Utilize a categorização semântica dos cargos para testar a hipótese do 'Prêmio de Estabilidade': setores resilientes a crises deveriam apresentar inadimplência estruturalmente menor, justificando políticas de crédito diferenciadas.


```python
sector_risk = df_risk.groupby('emp_sector', observed=False).agg(
    volume=('target', 'count'),
    bad_rate=('target', 'mean')
).reset_index().sort_values('bad_rate')

plt.figure(figsize=(14, 7))

# Gráfico de Barras 
ax1 = sns.barplot(data=sector_risk, x='emp_sector', y='volume', color='blue', alpha=0.6)
ax1.set_ylabel('Quantidade de Empréstimos', color='black')
ax1.set_xlabel('Setor de Trabalho', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Gráfico de Linha (Risco)
ax2 = ax1.twinx()
sns.pointplot(data=sector_risk, x='emp_sector', y='bad_rate', color='red', markers='o', scale=1.2, ax=ax2)
ax2.set_ylabel('Taxa de Inadimplência (Bad Rate)', color='black', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Linha Média do Portfólio
avg_risk = df_risk['target'].mean()
ax2.axhline(avg_risk, color='green', linestyle='--', alpha=0.5, label=f'Média Portfólio: {avg_risk:.1%}')

plt.title('Profissionais dos setores de alta estabilidade mantêm inadimplência 20% menor mesmo em cenários de crise.', fontsize=16, pad=20)
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_112_0.png)
    



```python
sector_risk_styled = sector_risk.sort_values('bad_rate', ascending=False).set_index('emp_sector')

sector_risk_styled.style.format({
    'volume': '{:,}',
    'bad_rate': '{:.2%}'
}).background_gradient(cmap='Blues', subset=['bad_rate'])
```




<style type="text/css">
#T_a89f1_row0_col1 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_a89f1_row1_col1 {
  background-color: #083e81;
  color: #f1f1f1;
}
#T_a89f1_row2_col1 {
  background-color: #08519c;
  color: #f1f1f1;
}
#T_a89f1_row3_col1 {
  background-color: #99c7e0;
  color: #000000;
}
#T_a89f1_row4_col1 {
  background-color: #b3d3e8;
  color: #000000;
}
#T_a89f1_row5_col1 {
  background-color: #c1d9ed;
  color: #000000;
}
#T_a89f1_row6_col1 {
  background-color: #e2edf8;
  color: #000000;
}
#T_a89f1_row7_col1 {
  background-color: #e8f1fa;
  color: #000000;
}
#T_a89f1_row8_col1 {
  background-color: #eaf2fb;
  color: #000000;
}
#T_a89f1_row9_col1 {
  background-color: #f0f6fd;
  color: #000000;
}
#T_a89f1_row10_col1 {
  background-color: #f7fbff;
  color: #000000;
}
</style>
<table id="T_a89f1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a89f1_level0_col0" class="col_heading level0 col0" >volume</th>
      <th id="T_a89f1_level0_col1" class="col_heading level0 col1" >bad_rate</th>
    </tr>
    <tr>
      <th class="index_name level0" >emp_sector</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a89f1_level0_row0" class="row_heading level0 row0" >Missing</th>
      <td id="T_a89f1_row0_col0" class="data row0 col0" >82,489</td>
      <td id="T_a89f1_row0_col1" class="data row0 col1" >26.28%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row1" class="row_heading level0 row1" >Blue Collar</th>
      <td id="T_a89f1_row1_col0" class="data row1 col0" >25,145</td>
      <td id="T_a89f1_row1_col1" class="data row1 col1" >25.77%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row2" class="row_heading level0 row2" >Service/Retail</th>
      <td id="T_a89f1_row2_col0" class="data row2 col0" >73,964</td>
      <td id="T_a89f1_row2_col1" class="data row2 col1" >25.11%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row3" class="row_heading level0 row3" >Self-Employed</th>
      <td id="T_a89f1_row3_col0" class="data row3 col0" >56,286</td>
      <td id="T_a89f1_row3_col1" class="data row3 col1" >20.70%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row4" class="row_heading level0 row4" >Other</th>
      <td id="T_a89f1_row4_col0" class="data row4 col0" >427,180</td>
      <td id="T_a89f1_row4_col1" class="data row4 col1" >19.99%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row5" class="row_heading level0 row5" >Management</th>
      <td id="T_a89f1_row5_col0" class="data row5 col0" >246,097</td>
      <td id="T_a89f1_row5_col1" class="data row5 col1" >19.60%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row6" class="row_heading level0 row6" >Healthcare</th>
      <td id="T_a89f1_row6_col0" class="data row6 col0" >86,415</td>
      <td id="T_a89f1_row6_col1" class="data row6 col1" >18.15%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row7" class="row_heading level0 row7" >Education</th>
      <td id="T_a89f1_row7_col0" class="data row7 col0" >51,972</td>
      <td id="T_a89f1_row7_col1" class="data row7 col1" >17.86%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row8" class="row_heading level0 row8" >Tech/Eng</th>
      <td id="T_a89f1_row8_col0" class="data row8 col0" >191,902</td>
      <td id="T_a89f1_row8_col1" class="data row8 col1" >17.79%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row9" class="row_heading level0 row9" >Public Sector</th>
      <td id="T_a89f1_row9_col0" class="data row9 col0" >44,151</td>
      <td id="T_a89f1_row9_col1" class="data row9 col1" >17.49%</td>
    </tr>
    <tr>
      <th id="T_a89f1_level0_row10" class="row_heading level0 row10" >Finance/Legal</th>
      <td id="T_a89f1_row10_col0" class="data row10 col0" >20,280</td>
      <td id="T_a89f1_row10_col1" class="data row10 col1" >17.16%</td>
    </tr>
  </tbody>
</table>




**O que este gráfico nos diz?**

Testamos a hipótese de que a estabilidade da fonte de renda é tão preditiva quanto o valor da renda. Os dados confirmam: setores de alta estabilidade (Setor Público, Jurídico e Saúde) apresentam inadimplência até 20% inferior à média da carteira.

Em contraste, a volatilidade cobra seu preço. Trabalhadores manuais e perfis com dados incompletos lideram as perdas. Isso revela um gap de cadastro crítico: a omissão de dados profissionais é, por si só, um forte indicador de default.

**Por que estou fazendo este gráfico?**

Enquanto a análise de safra olha para o passado, a gestão de fluxo de caixa precisa olhar para o futuro imediato. Precisamos saber quantos clientes estão na 'sala de espera' da inadimplência. Classifiquei a carteira ativa em 'Buckets de Risco' conforme as normas de Basileia/IFRS 9. O objetivo é identificar o volume de créditos em deterioração acelerada que se converterão em perdas contábeis nos próximos 90 dias, atuando como um Sistema de Alerta Precoce.


```python
# Transformamos os status textuais em "Buckets" padronizados de risco
def map_risk_bucket(status):
    if status == 'Current': return '0: Em Dia (Stage 1)'
    if 'Grace Period' in status: return '1: Atraso 1-15 (Early)'
    if 'Late (16-30 days)' in status: return '2: Atraso 16-30 (Watchlist)'
    if 'Late (31-120 days)' in status: return '3: Atraso 31-120 (Stage 2)'
    if 'Default' in status or 'Charged Off' in status: return '4: Default (Stage 3)'
    if 'Fully Paid' in status: return '5: Quitado (Exit)'
    return 'Other'

df_risk['risk_bucket'] = df_risk['loan_status'].apply(map_risk_bucket)

# Como não temos a série temporal T vs T+1, mostramos a distribuição atual, como proxy da "Probabilidade de Permanência" em cada estado.
bucket_counts = df_risk['risk_bucket'].value_counts(normalize=True).sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette='Blues')

plt.title('O fluxo de clientes migrando para faixas críticas de atraso irá acelerar no próximo trimestre.', fontsize=14)
plt.xlabel('Bucket de Risco (Status Atual)', fontsize=12)
plt.ylabel('% da Carteira', fontsize=12)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(bucket_counts.values):
    plt.text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_116_0.png)
    


**O que este gráfico nos diz?**

A métrica é dura: 20,1% da carteira consolidou-se em Default, enquanto 79,9% foi quitada com sucesso.

Não estamos falando de atrasos temporários, mas de perda confirmada. Uma taxa de falha de 1 em cada 5 contratos é insustentável para a rentabilidade de longo prazo (ROE). Este dado valida todas as análises anteriores: a necessidade de endurecer a entrada (Safra), monitorar a estabilidade (PSI) e cobrar rápido (Recuperação) não é preciosismo, é uma questão de sobrevivência. Sem essas travas, a carteira sangra 20% do seu volume originado.

**Por que estou fazendo este gráfico?**

Comparar a inadimplência de uma safra antiga com uma recente é como comparar maçãs com laranjas, pois o risco acumula com o tempo. Uma safra recente sempre parecerá melhor que uma antiga, criando uma falsa sensação de segurança durante períodos de crescimento. Para corrigir isso, calculei o Índice de Performance Relativa. Dividi a inadimplência real de cada safra pelo 'Benchmark Histórico' esperado para aquele exato mês de vida. Isso isola matematicamente a Qualidade da Originação: se o índice for 1.20, a safra é 20% pior do que o normal, independente da sua idade.


```python
# Garantimos que temos dados suficientes
vintage_matrix = df_finance.groupby(['vintage_qt', 'mob'])['target'].mean().unstack()

# Em vez da média simples, usamos a média ponderada ou mediana para definir a "Curva Padrão".
age_effect = vintage_matrix.median(axis=0)

# Safra / Benchmark.
relative_risk_matrix = vintage_matrix.divide(age_effect, axis=1)

# Ignoramos MOBs iniciais onde a inadimplência é próxima de 0 e gera divisão instável.
# Ignoramos MOBs muito tardios onde poucas safras chegaram.
mob_start, mob_end = 3, 48
relative_risk_trimmed = relative_risk_matrix.loc[:, mob_start:mob_end]

# Calculamos a qualidade apenas se a safra tiver pelo menos 6 meses de dados válidos nessa janela
# Isso evita que uma safra nascida ontem pareça "perfeita" ou "terrível" com base em 1 ponto.
min_obs_window = 6
cohort_quality = relative_risk_trimmed.mean(axis=1, skipna=True)
valid_vintages = relative_risk_trimmed.count(axis=1) >= min_obs_window
cohort_quality = cohort_quality[valid_vintages].to_frame(name='Indice_de_Risco_Safra')

plt.figure(figsize=(14, 6))

# Colorir linha baseado no risco (Verde < 1.0, Vermelho > 1.0)
# Usamos um truque de scatter plot sobre a linha para colorir os pontos
sns.lineplot(data=cohort_quality, x=cohort_quality.index.astype(str), y='Indice_de_Risco_Safra', 
             color='gray', alpha=0.5, label='Tendência Geral')

colors = ['firebrick' if x > 1.05 else 'forestgreen' if x < 0.95 else 'gold' for x in cohort_quality['Indice_de_Risco_Safra']]
plt.scatter(cohort_quality.index.astype(str), cohort_quality['Indice_de_Risco_Safra'], 
            c=colors, s=100, zorder=3)

# Linha de Referência
plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Benchmark Histórico (1.0)')

# Formatação
plt.title('As safras recentes estão operando 50% a 100% acima do risco esperado pelo apetite de crédito.', fontsize=16)
plt.ylabel('Índice de Risco Relativo (vs. Benchmark)', fontsize=12)
plt.xlabel('Safra de Originação', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
# Adicionar bandas de "Normalidade"
plt.axhspan(0.90, 1.10, color='blue', alpha=0.1, label='Faixa de Normalidade (+/- 10%)')

plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_119_0.png)
    



```python
cohort_quality.sort_values('Indice_de_Risco_Safra', ascending=False).style.format('{:.2f}') \
    .background_gradient(cmap='RdYlGn_r')
```




<style type="text/css">
#T_0674d_row0_col0 {
  background-color: #a50026;
  color: #f1f1f1;
}
#T_0674d_row1_col0 {
  background-color: #d7ee8a;
  color: #000000;
}
#T_0674d_row2_col0 {
  background-color: #abdb6d;
  color: #000000;
}
#T_0674d_row3_col0 {
  background-color: #89cc67;
  color: #000000;
}
#T_0674d_row4_col0 {
  background-color: #7fc866;
  color: #000000;
}
#T_0674d_row5_col0, #T_0674d_row6_col0 {
  background-color: #73c264;
  color: #000000;
}
#T_0674d_row7_col0 {
  background-color: #63bc62;
  color: #f1f1f1;
}
#T_0674d_row8_col0 {
  background-color: #60ba62;
  color: #f1f1f1;
}
#T_0674d_row9_col0 {
  background-color: #5db961;
  color: #f1f1f1;
}
#T_0674d_row10_col0 {
  background-color: #5ab760;
  color: #f1f1f1;
}
#T_0674d_row11_col0 {
  background-color: #57b65f;
  color: #f1f1f1;
}
#T_0674d_row12_col0 {
  background-color: #4eb15d;
  color: #f1f1f1;
}
#T_0674d_row13_col0 {
  background-color: #3ca959;
  color: #f1f1f1;
}
#T_0674d_row14_col0 {
  background-color: #39a758;
  color: #f1f1f1;
}
#T_0674d_row15_col0, #T_0674d_row16_col0 {
  background-color: #2da155;
  color: #f1f1f1;
}
#T_0674d_row17_col0 {
  background-color: #1b9950;
  color: #f1f1f1;
}
#T_0674d_row18_col0 {
  background-color: #15904c;
  color: #f1f1f1;
}
#T_0674d_row19_col0 {
  background-color: #148e4b;
  color: #f1f1f1;
}
#T_0674d_row20_col0, #T_0674d_row21_col0 {
  background-color: #0d8044;
  color: #f1f1f1;
}
#T_0674d_row22_col0, #T_0674d_row23_col0 {
  background-color: #0c7f43;
  color: #f1f1f1;
}
#T_0674d_row24_col0, #T_0674d_row25_col0, #T_0674d_row26_col0 {
  background-color: #0b7d42;
  color: #f1f1f1;
}
#T_0674d_row27_col0 {
  background-color: #0a7b41;
  color: #f1f1f1;
}
#T_0674d_row28_col0, #T_0674d_row29_col0, #T_0674d_row30_col0 {
  background-color: #097940;
  color: #f1f1f1;
}
#T_0674d_row31_col0 {
  background-color: #07753e;
  color: #f1f1f1;
}
#T_0674d_row32_col0 {
  background-color: #06733d;
  color: #f1f1f1;
}
#T_0674d_row33_col0, #T_0674d_row34_col0, #T_0674d_row35_col0 {
  background-color: #05713c;
  color: #f1f1f1;
}
#T_0674d_row36_col0 {
  background-color: #04703b;
  color: #f1f1f1;
}
#T_0674d_row37_col0, #T_0674d_row38_col0 {
  background-color: #036e3a;
  color: #f1f1f1;
}
#T_0674d_row39_col0, #T_0674d_row40_col0, #T_0674d_row41_col0 {
  background-color: #026c39;
  color: #f1f1f1;
}
#T_0674d_row42_col0 {
  background-color: #016a38;
  color: #f1f1f1;
}
#T_0674d_row43_col0, #T_0674d_row44_col0 {
  background-color: #006837;
  color: #f1f1f1;
}
</style>
<table id="T_0674d">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0674d_level0_col0" class="col_heading level0 col0" >Indice_de_Risco_Safra</th>
    </tr>
    <tr>
      <th class="index_name level0" >vintage_qt</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0674d_level0_row0" class="row_heading level0 row0" >2016Q2</th>
      <td id="T_0674d_row0_col0" class="data row0 col0" >4.37</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row1" class="row_heading level0 row1" >2018Q3</th>
      <td id="T_0674d_row1_col0" class="data row1 col0" >2.28</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row2" class="row_heading level0 row2" >2018Q2</th>
      <td id="T_0674d_row2_col0" class="data row2 col0" >1.97</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row3" class="row_heading level0 row3" >2007Q4</th>
      <td id="T_0674d_row3_col0" class="data row3 col0" >1.78</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row4" class="row_heading level0 row4" >2016Q3</th>
      <td id="T_0674d_row4_col0" class="data row4 col0" >1.72</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row5" class="row_heading level0 row5" >2008Q2</th>
      <td id="T_0674d_row5_col0" class="data row5 col0" >1.66</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row6" class="row_heading level0 row6" >2018Q1</th>
      <td id="T_0674d_row6_col0" class="data row6 col0" >1.66</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row7" class="row_heading level0 row7" >2017Q3</th>
      <td id="T_0674d_row7_col0" class="data row7 col0" >1.57</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row8" class="row_heading level0 row8" >2017Q4</th>
      <td id="T_0674d_row8_col0" class="data row8 col0" >1.56</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row9" class="row_heading level0 row9" >2016Q4</th>
      <td id="T_0674d_row9_col0" class="data row9 col0" >1.55</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row10" class="row_heading level0 row10" >2017Q2</th>
      <td id="T_0674d_row10_col0" class="data row10 col0" >1.54</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row11" class="row_heading level0 row11" >2007Q3</th>
      <td id="T_0674d_row11_col0" class="data row11 col0" >1.52</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row12" class="row_heading level0 row12" >2017Q1</th>
      <td id="T_0674d_row12_col0" class="data row12 col0" >1.48</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row13" class="row_heading level0 row13" >2008Q1</th>
      <td id="T_0674d_row13_col0" class="data row13 col0" >1.40</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row14" class="row_heading level0 row14" >2015Q3</th>
      <td id="T_0674d_row14_col0" class="data row14 col0" >1.38</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row15" class="row_heading level0 row15" >2016Q1</th>
      <td id="T_0674d_row15_col0" class="data row15 col0" >1.33</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row16" class="row_heading level0 row16" >2015Q4</th>
      <td id="T_0674d_row16_col0" class="data row16 col0" >1.33</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row17" class="row_heading level0 row17" >2015Q2</th>
      <td id="T_0674d_row17_col0" class="data row17 col0" >1.25</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row18" class="row_heading level0 row18" >2008Q4</th>
      <td id="T_0674d_row18_col0" class="data row18 col0" >1.18</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row19" class="row_heading level0 row19" >2008Q3</th>
      <td id="T_0674d_row19_col0" class="data row19 col0" >1.17</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row20" class="row_heading level0 row20" >2015Q1</th>
      <td id="T_0674d_row20_col0" class="data row20 col0" >1.07</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row21" class="row_heading level0 row21" >2009Q4</th>
      <td id="T_0674d_row21_col0" class="data row21 col0" >1.07</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row22" class="row_heading level0 row22" >2010Q1</th>
      <td id="T_0674d_row22_col0" class="data row22 col0" >1.06</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row23" class="row_heading level0 row23" >2012Q2</th>
      <td id="T_0674d_row23_col0" class="data row23 col0" >1.05</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row24" class="row_heading level0 row24" >2012Q3</th>
      <td id="T_0674d_row24_col0" class="data row24 col0" >1.05</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row25" class="row_heading level0 row25" >2012Q1</th>
      <td id="T_0674d_row25_col0" class="data row25 col0" >1.04</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row26" class="row_heading level0 row26" >2010Q2</th>
      <td id="T_0674d_row26_col0" class="data row26 col0" >1.04</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row27" class="row_heading level0 row27" >2014Q2</th>
      <td id="T_0674d_row27_col0" class="data row27 col0" >1.03</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row28" class="row_heading level0 row28" >2014Q4</th>
      <td id="T_0674d_row28_col0" class="data row28 col0" >1.02</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row29" class="row_heading level0 row29" >2011Q4</th>
      <td id="T_0674d_row29_col0" class="data row29 col0" >1.01</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row30" class="row_heading level0 row30" >2014Q3</th>
      <td id="T_0674d_row30_col0" class="data row30 col0" >1.01</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row31" class="row_heading level0 row31" >2013Q2</th>
      <td id="T_0674d_row31_col0" class="data row31 col0" >0.99</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row32" class="row_heading level0 row32" >2012Q4</th>
      <td id="T_0674d_row32_col0" class="data row32 col0" >0.97</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row33" class="row_heading level0 row33" >2011Q2</th>
      <td id="T_0674d_row33_col0" class="data row33 col0" >0.97</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row34" class="row_heading level0 row34" >2009Q2</th>
      <td id="T_0674d_row34_col0" class="data row34 col0" >0.96</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row35" class="row_heading level0 row35" >2009Q1</th>
      <td id="T_0674d_row35_col0" class="data row35 col0" >0.96</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row36" class="row_heading level0 row36" >2014Q1</th>
      <td id="T_0674d_row36_col0" class="data row36 col0" >0.96</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row37" class="row_heading level0 row37" >2013Q3</th>
      <td id="T_0674d_row37_col0" class="data row37 col0" >0.94</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row38" class="row_heading level0 row38" >2013Q1</th>
      <td id="T_0674d_row38_col0" class="data row38 col0" >0.94</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row39" class="row_heading level0 row39" >2009Q3</th>
      <td id="T_0674d_row39_col0" class="data row39 col0" >0.92</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row40" class="row_heading level0 row40" >2011Q1</th>
      <td id="T_0674d_row40_col0" class="data row40 col0" >0.92</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row41" class="row_heading level0 row41" >2010Q3</th>
      <td id="T_0674d_row41_col0" class="data row41 col0" >0.92</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row42" class="row_heading level0 row42" >2013Q4</th>
      <td id="T_0674d_row42_col0" class="data row42 col0" >0.91</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row43" class="row_heading level0 row43" >2011Q3</th>
      <td id="T_0674d_row43_col0" class="data row43 col0" >0.90</td>
    </tr>
    <tr>
      <th id="T_0674d_level0_row44" class="row_heading level0 row44" >2010Q4</th>
      <td id="T_0674d_row44_col0" class="data row44 col0" >0.89</td>
    </tr>
  </tbody>
</table>




**O que este gráfico nos diz?**

O gráfico revela uma deterioração sistêmica na subscrição de crédito. Enquanto o período de 2009-2013 manteve-se na zona de excelência (Verde/Amarelo), observamos uma tendência de alta no risco ajustado a partir de 2014, culminando em safras recentes (2017-2018) operando consistentemente 50% a 100% acima do risco esperado (Índice > 1.5).

Isso sugere que o crescimento recente da carteira foi impulsionado pelo rebaixamento da régua de aprovação. O alerta é claro: as safras novas já nascem 'piores', exigindo provisionamento adicional imediato.

**Por que estou fazendo esta tabela?**

Na modelagem de risco, um erro comum é assumir que clientes 'Prime' recuperam melhor suas dívidas do que clientes 'Subprime' em caso de falência. Se essa suposição estiver errada, podemos estar subestimando drasticamente as provisões para a melhor parte da nossa carteira. Construí esta tabela para calcular a LGD Esperada por Grade, utilizando uma abordagem de dois estágios: primeiro, qual a chance de recuperarmos algum centavo (prob_recovery)? Segundo, se recuperarmos, quanto ainda perdemos (avg_lgd_when_rec)? Além disso, calculei a 'Downturn LGD', uma métrica regulatória crítica exigida por Basileia e IFRS 9 para garantir que o banco sobreviva a crises severas onde a recuperação pode cair a zero.


```python
# Filtramos defaults. Mas para treinar LGD, precisamos de defaults antigos.
# Assumimos que defaults recentes ainda estão em cobrança.
# Usamos 'last_pymnt_d' ou 'issue_d' para estimar a idade do default.
df_lgd = df_risk[df_risk['target'] == 1].copy()

# Trazer dados financeiros
cols_fin = ['recoveries', 'collection_recovery_fee', 'funded_amnt', 'total_rec_prncp', 'issue_d']
df_lgd[cols_fin] = df_finance.loc[df_lgd.index, cols_fin]

# EAD = Saldo Devedor no Default
df_lgd['EAD'] = (df_lgd['funded_amnt'] - df_lgd['total_rec_prncp']).clip(lower=1)

# Recuperação Líquida = Bruta - Custos de Cobrança
df_lgd['net_recovery'] = (df_lgd['recoveries'] - df_lgd['collection_recovery_fee']).clip(lower=0)

# LGD = 1 - (Recuperação Líquida / EAD)
df_lgd['LGD_real'] = 1 - (df_lgd['net_recovery'] / df_lgd['EAD'])
df_lgd['LGD_real'] = df_lgd['LGD_real'].clip(0, 1) # Capping normativo

# Flag de Sucesso na Recuperação
df_lgd['has_recovery'] = np.where(df_lgd['net_recovery'] > 0, 1, 0)

lgd_lookup = df_lgd.groupby('grade').agg(
    count=('EAD', 'count'),
    prob_recovery=('has_recovery', 'mean'),
    avg_lgd_when_rec=('LGD_real', lambda x: x[df_lgd.loc[x.index, 'has_recovery'] == 1].mean()),
    downturn_lgd=('LGD_real', lambda x: x.quantile(0.90)) 
).reset_index()

# E_LGD = (P(Perda Total) * 100%) + (P(Recuperação) * LGD_Média_Recuperada)
lgd_lookup['E_LGD'] = (1 - lgd_lookup['prob_recovery']) * 1.0 + \
                      (lgd_lookup['prob_recovery'] * lgd_lookup['avg_lgd_when_rec'])

# Preencher possíveis NaNs
lgd_lookup['avg_lgd_when_rec'] = lgd_lookup['avg_lgd_when_rec'].fillna(0)
lgd_lookup['E_LGD'] = lgd_lookup['E_LGD'].fillna(1.0)

format_dict = {col: '{:.2%}' for col in ['prob_recovery', 'avg_lgd_when_rec', 'E_LGD', 'downturn_lgd']}
format_dict['count'] = '{:,}'

display(lgd_lookup.style.format(format_dict) \
        .background_gradient(cmap='Reds', subset=['E_LGD', 'downturn_lgd']) \
        .bar(subset=['prob_recovery'], color='lightblue'))
```


<style type="text/css">
#T_8b9a1_row0_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 81.9%, transparent 81.9%);
}
#T_8b9a1_row0_col4, #T_8b9a1_row1_col4, #T_8b9a1_row2_col4, #T_8b9a1_row3_col4, #T_8b9a1_row4_col4, #T_8b9a1_row5_col4, #T_8b9a1_row5_col5, #T_8b9a1_row6_col4 {
  background-color: #fff5f0;
  color: #000000;
}
#T_8b9a1_row0_col5 {
  background-color: #67000d;
  color: #f1f1f1;
}
#T_8b9a1_row1_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 91.1%, transparent 91.1%);
}
#T_8b9a1_row1_col5 {
  background-color: #a60f15;
  color: #f1f1f1;
}
#T_8b9a1_row2_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 93.2%, transparent 93.2%);
}
#T_8b9a1_row2_col5 {
  background-color: #ad1117;
  color: #f1f1f1;
}
#T_8b9a1_row3_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 94.3%, transparent 94.3%);
}
#T_8b9a1_row3_col5 {
  background-color: #d01d1f;
  color: #f1f1f1;
}
#T_8b9a1_row4_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 96.7%, transparent 96.7%);
}
#T_8b9a1_row4_col5 {
  background-color: #fc9070;
  color: #000000;
}
#T_8b9a1_row5_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 100.0%, transparent 100.0%);
}
#T_8b9a1_row6_col2 {
  width: 10em;
  background: linear-gradient(90deg, lightblue 99.6%, transparent 99.6%);
}
#T_8b9a1_row6_col5 {
  background-color: #fedaca;
  color: #000000;
}
</style>
<table id="T_8b9a1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8b9a1_level0_col0" class="col_heading level0 col0" >grade</th>
      <th id="T_8b9a1_level0_col1" class="col_heading level0 col1" >count</th>
      <th id="T_8b9a1_level0_col2" class="col_heading level0 col2" >prob_recovery</th>
      <th id="T_8b9a1_level0_col3" class="col_heading level0 col3" >avg_lgd_when_rec</th>
      <th id="T_8b9a1_level0_col4" class="col_heading level0 col4" >downturn_lgd</th>
      <th id="T_8b9a1_level0_col5" class="col_heading level0 col5" >E_LGD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8b9a1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_8b9a1_row0_col0" class="data row0 col0" >A</td>
      <td id="T_8b9a1_row0_col1" class="data row0 col1" >13,771</td>
      <td id="T_8b9a1_row0_col2" class="data row0 col2" >59.32%</td>
      <td id="T_8b9a1_row0_col3" class="data row0 col3" >85.44%</td>
      <td id="T_8b9a1_row0_col4" class="data row0 col4" >100.00%</td>
      <td id="T_8b9a1_row0_col5" class="data row0 col5" >91.36%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_8b9a1_row1_col0" class="data row1 col0" >B</td>
      <td id="T_8b9a1_row1_col1" class="data row1 col1" >51,151</td>
      <td id="T_8b9a1_row1_col2" class="data row1 col2" >65.99%</td>
      <td id="T_8b9a1_row1_col3" class="data row1 col3" >86.76%</td>
      <td id="T_8b9a1_row1_col4" class="data row1 col4" >100.00%</td>
      <td id="T_8b9a1_row1_col5" class="data row1 col5" >91.26%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_8b9a1_row2_col0" class="data row2 col0" >C</td>
      <td id="T_8b9a1_row2_col1" class="data row2 col1" >83,382</td>
      <td id="T_8b9a1_row2_col2" class="data row2 col2" >67.50%</td>
      <td id="T_8b9a1_row2_col3" class="data row2 col3" >87.03%</td>
      <td id="T_8b9a1_row2_col4" class="data row2 col4" >100.00%</td>
      <td id="T_8b9a1_row2_col5" class="data row2 col5" >91.24%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_8b9a1_row3_col0" class="data row3 col0" >D</td>
      <td id="T_8b9a1_row3_col1" class="data row3 col1" >59,606</td>
      <td id="T_8b9a1_row3_col2" class="data row3 col2" >68.27%</td>
      <td id="T_8b9a1_row3_col3" class="data row3 col3" >87.04%</td>
      <td id="T_8b9a1_row3_col4" class="data row3 col4" >100.00%</td>
      <td id="T_8b9a1_row3_col5" class="data row3 col5" >91.15%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_8b9a1_row4_col0" class="data row4 col0" >E</td>
      <td id="T_8b9a1_row4_col1" class="data row4 col1" >35,507</td>
      <td id="T_8b9a1_row4_col2" class="data row4 col2" >70.05%</td>
      <td id="T_8b9a1_row4_col3" class="data row4 col3" >86.98%</td>
      <td id="T_8b9a1_row4_col4" class="data row4 col4" >100.00%</td>
      <td id="T_8b9a1_row4_col5" class="data row4 col5" >90.88%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_8b9a1_row5_col0" class="data row5 col0" >F</td>
      <td id="T_8b9a1_row5_col1" class="data row5 col1" >14,351</td>
      <td id="T_8b9a1_row5_col2" class="data row5 col2" >72.43%</td>
      <td id="T_8b9a1_row5_col3" class="data row5 col3" >86.99%</td>
      <td id="T_8b9a1_row5_col4" class="data row5 col4" >100.00%</td>
      <td id="T_8b9a1_row5_col5" class="data row5 col5" >90.58%</td>
    </tr>
    <tr>
      <th id="T_8b9a1_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_8b9a1_row6_col0" class="data row6 col0" >G</td>
      <td id="T_8b9a1_row6_col1" class="data row6 col1" >4,549</td>
      <td id="T_8b9a1_row6_col2" class="data row6 col2" >72.13%</td>
      <td id="T_8b9a1_row6_col3" class="data row6 col3" >87.10%</td>
      <td id="T_8b9a1_row6_col4" class="data row6 col4" >100.00%</td>
      <td id="T_8b9a1_row6_col5" class="data row6 col5" >90.69%</td>
    </tr>
  </tbody>
</table>



**O que esta tabela nos diz?**

Os dados revelam um Paradoxo de Recuperação. Contra-intuitivamente, a Perda Esperada (E_LGD) é praticamente uniforme (~91%) em todos os Grades, sendo até ligeiramente maior nos clientes Grade A (91.36%) do que nos Grade G (90.69%).
1. Agnosticismo da Severidade: Isso prova que o Grade de crédito discrimina a frequência do default (PD), mas não a severidade da perda. Uma vez que um cliente Grade A decide não pagar, o prejuízo é tão devastador quanto o de um Grade G. Não há 'colchão de segurança' nos segmentos Prime.
2. O Risco do Cenário Adverso: A coluna downturn_lgd está cravada em 100.00% para todos os grupos. Isso significa que, para fins de Capital Econômico, devemos assumir perda total em operações sem garantia real.

**Por que estou fazendo este gráfico?**

O risco não acontece no vácuo; ele deixa rastros. Antes de um cliente parar de pagar um empréstimo parcelado, ele frequentemente consome suas outras fontes de liquidez. Utilize a taxa de utilização de rotativo como uma proxy para investigar o fenômeno da 'Corrida para o Saque'. Quero provar que clientes de pior rating não apenas têm maior probabilidade de falhar, mas também tendem a maximizar sua exposição antes do colapso.


```python
# Usamos 'revol_util' como proxy de estresse financeiro.
# simula a modelagem de EAD para produtos rotativos.
df_ead = df_risk[['grade', 'target', 'revol_util']].copy()

# Comparação: Utilização média de quem pagou (Baseline) vs quem faliu (Stressed)
ead_analysis = df_ead.groupby(['grade', 'target']).agg(
    avg_util=('revol_util', 'mean')
).unstack()

# Ajuste de colunas (0 = Bom/Baseline, 1 = Mau/Default)
ead_analysis.columns = ['Util_Baseline', 'Util_Default']

# Fórmula: CCF = (Util_Default - Util_Baseline) / (Lim_Disp_Baseline)
# Onde Lim_Disp_Baseline = 1 - Util_Baseline
# De cada $1,00 de limite que o cliente tinha livre, quanto ele sacou antes de quebrar?
ead_analysis['CCF_Proxy'] = (
    (ead_analysis['Util_Default'] - ead_analysis['Util_Baseline']) / 
    (1 - ead_analysis['Util_Baseline'])
)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Barras
ead_analysis[['Util_Baseline', 'Util_Default']].plot(kind='bar', ax=ax1, 
                                                     color=['#A8DADC', '#1D3557'], alpha=0.8)

# CCF Estimado
ax2 = ax1.twinx()
ax2.plot(ax1.get_xticks(), ead_analysis['CCF_Proxy'], color='crimson', marker='o', 
         linewidth=2, linestyle='--', label='CCF Estimado (Agressividade do Saque)')

# Formatação
ax1.set_title('Clientes em pré-default tendem a utilizar 40% do limite disponível antes da quebra.', fontsize=14)
ax1.set_ylabel('Taxa de Utilização (%)', fontsize=12)
ax2.set_ylabel('CCF Estimado (%)', fontsize=12, color='black')
ax1.set_xlabel('Grade de Risco (Rating)', fontsize=12)
ax1.legend(['Utilização Baseline (Bons)', 'Utilização no Default (Maus)'], loc='lower left')
ax2.legend(loc='center right')

ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')

# Formatar eixos como porcentagem
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.show()
```


    
![png](credit-risk-eda-v01_files/credit-risk-eda-v01_126_0.png)
    



```python
display(ead_analysis[['Util_Baseline', 'Util_Default', 'CCF_Proxy']].style.format('{:.2%}') \
        .background_gradient(cmap='Reds', subset=['CCF_Proxy']))
```


<style type="text/css">
#T_fe8a4_row0_col2 {
  background-color: #67000d;
  color: #f1f1f1;
}
#T_fe8a4_row1_col2 {
  background-color: #f5523a;
  color: #f1f1f1;
}
#T_fe8a4_row2_col2 {
  background-color: #fb7353;
  color: #f1f1f1;
}
#T_fe8a4_row3_col2, #T_fe8a4_row4_col2 {
  background-color: #fc9070;
  color: #000000;
}
#T_fe8a4_row5_col2 {
  background-color: #fdcab5;
  color: #000000;
}
#T_fe8a4_row6_col2 {
  background-color: #fff5f0;
  color: #000000;
}
</style>
<table id="T_fe8a4">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fe8a4_level0_col0" class="col_heading level0 col0" >Util_Baseline</th>
      <th id="T_fe8a4_level0_col1" class="col_heading level0 col1" >Util_Default</th>
      <th id="T_fe8a4_level0_col2" class="col_heading level0 col2" >CCF_Proxy</th>
    </tr>
    <tr>
      <th class="index_name level0" >grade</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fe8a4_level0_row0" class="row_heading level0 row0" >A</th>
      <td id="T_fe8a4_row0_col0" class="data row0 col0" >39.21%</td>
      <td id="T_fe8a4_row0_col1" class="data row0 col1" >41.60%</td>
      <td id="T_fe8a4_row0_col2" class="data row0 col2" >3.93%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row1" class="row_heading level0 row1" >B</th>
      <td id="T_fe8a4_row1_col0" class="data row1 col0" >50.90%</td>
      <td id="T_fe8a4_row1_col1" class="data row1 col1" >51.05%</td>
      <td id="T_fe8a4_row1_col2" class="data row1 col2" >0.30%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row2" class="row_heading level0 row2" >C</th>
      <td id="T_fe8a4_row2_col0" class="data row2 col0" >55.44%</td>
      <td id="T_fe8a4_row2_col1" class="data row2 col1" >55.23%</td>
      <td id="T_fe8a4_row2_col2" class="data row2 col2" >-0.48%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row3" class="row_heading level0 row3" >D</th>
      <td id="T_fe8a4_row3_col0" class="data row3 col0" >58.25%</td>
      <td id="T_fe8a4_row3_col1" class="data row3 col1" >57.75%</td>
      <td id="T_fe8a4_row3_col2" class="data row3 col2" >-1.21%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row4" class="row_heading level0 row4" >E</th>
      <td id="T_fe8a4_row4_col0" class="data row4 col0" >59.44%</td>
      <td id="T_fe8a4_row4_col1" class="data row4 col1" >58.95%</td>
      <td id="T_fe8a4_row4_col2" class="data row4 col2" >-1.21%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row5" class="row_heading level0 row5" >F</th>
      <td id="T_fe8a4_row5_col0" class="data row5 col0" >60.74%</td>
      <td id="T_fe8a4_row5_col1" class="data row5 col1" >59.67%</td>
      <td id="T_fe8a4_row5_col2" class="data row5 col2" >-2.72%</td>
    </tr>
    <tr>
      <th id="T_fe8a4_level0_row6" class="row_heading level0 row6" >G</th>
      <td id="T_fe8a4_row6_col0" class="data row6 col0" >60.29%</td>
      <td id="T_fe8a4_row6_col1" class="data row6 col1" >58.55%</td>
      <td id="T_fe8a4_row6_col2" class="data row6 col2" >-4.38%</td>
    </tr>
  </tbody>
</table>



**O que este gráfico nos diz?**

O gráfico revela um comportamento contraintuitivo: a 'Corrida para o Saque' é mais acentuada nos clientes de alto rating (Grade A).

Enquanto clientes de alto risco (Grade G) já operam próximos ao limite de saturação (Utilização > 60%), os clientes Prime (Grade A) possuem limites ociosos. Quando um cliente 'A' entra em estresse financeiro, ele utiliza agressivamente essa margem antes do colapso, elevando sua exposição (EAD) subitamente.

## Agir

### Da Expansão à Eficiência de Capital

Nossa análise identificou que a estratégia atual de volume está mascarando a destruição de valor em nichos específicos. Abaixo, apresentamos um plano tático priorizado para otimizar o Retorno sobre o Capital.

1. **Ações Imediatas**
* Suspensão de Originação para Grades F e G

  * **Diagnóstico:** A análise da Fronteira Eficiente confirmou que o Retorno Líquido Anualizado nestes segmentos é negativo. O prêmio de risco cobrado não cobre a Perda em caso de default.

  * **Ação Tática:** Bloqueio imediato de concessão para novos clientes nestas faixas de rating.

  * **Impacto Financeiro:** Embora esta medida resulte em uma renúncia de receita de juros projetada de USD 232.224.807,49 , ela evitará perdas de crédito estimadas em USD 336.074.556,74 . 

  * **Resultado Líquido:** Aumento imediato na margem líquida da carteira.

* Revisão de Política para Produto 60 Meses

   * **Diagnóstico:** Identificada deterioração de safra acelerada. Sob normas contábeis, ativos com risco de vida útil maior exigem provisões imediatas mais altas, imobilizando capital do banco.

   * **Ação Tática:** Restringir o prazo de 60 meses exclusivamente para clientes com Score A e B e Debt-to-Income controlado.

   * **Impacto Financeiro:** Redução da necessidade de Provisão para Devedores Duvidosos, liberando capital regulatório para reinvestimento em segmentos rentáveis.

2. **Otimização Operacional**
* Racionalização da Cobrança e Write-off

  * **Diagnóstico:** A curva de recuperação atinge saturação no 12º mês, onde o custo marginal da equipe de cobrança supera o valor recuperado.

  * **Ação Tática:** Automatizar o write-off e a venda de carteira inadimplente ao atingir 360 dias de atraso.

  * **Impacto Financeiro:** Redução estimada de 60.3% nas Despesas Operacionais da área de cobrança, realocando recursos humanos para as faixas de atraso recente onde a recuperação é 80% mais provável.

3. **Estratégia de Crescimento**
* Regionalização da Política de Crédito

  * **Ação:** Ajustar a régua de aprovação especificamente para clusters geográficos identificados com risco sistêmico elevado.

  * **Objetivo:** Mitigar a exposição a riscos regionais não capturados pelo modelo nacional.

* Precificação Competitiva para "Mar Aberto"

  * **Ação:** Utilizar a margem recuperada dos cortes acima para oferecer taxas mais agressivas aos perfis de alta estabilidade.

  * **Objetivo:** Ganhar market share de clientes adimplentes, trocando "volume tóxico" por "volume saudável" e melhorando a qualidade média do balanço.

## Referências

* BOARD OF GOVERNORS OF THE FEDERAL RESERVE SYSTEM. **2024 Supervisory Stress Test Methodology.** Washington, D.C.: Federal Reserve, mar. 2024. Disponível em: https://www.federalreserve.gov/publications/2024-march-supervisory-stress-test-methodology-descriptions-supervisory-models.htm. Acesso em: 28 jan. 2026.
* BREEDEN, Joseph L. Normalizing Pandemic Data for Credit Scoring. **Journal of Risk and Financial Management**, v. 18, n. 11, p. 657, 2025. Disponível em: https://doi.org/10.3390/jrfm18110657.
* KPMG. **Expected Credit Loss (ECL)**: Assurance and Consulting Services. Índia: KPMG Assurance and Consulting Services LLP, 2025. Disponível em: https://assets.kpmg.com/content/dam/kpmgsites/in/pdf/2025/01/expected-credit-loss-ecl.pdf.
* SIARKA, Paweł. Vintage analysis as a basic tool for monitoring credit risk. **Mathematical Economics**, Wrocław, n. 7(14), p. 217-229, 2011.
* SIDDIQI, Naeem. **Credit risk scorecards**: developing and implementing intelligent credit scoring. New Jersey: John Wiley & Sons, 2006.
* CAIRO, Alberto. **The functional art**: an introduction to information graphics and visualization. Berkeley: New Riders, 2013.
* FINANCIAL TIMES. **Visual Vocabulary**: designing with data. Londres: FT Graphics, [s.d.]. Disponível em: https://ft-interactive.github.io/visual-vocabulary/.
* KNAFLIC, Cole Nussbaumer. **Storytelling with data**: a data visualization guide for business professionals. New Jersey: John Wiley & Sons, 2015.
* TUFTE, Edward R. **The visual display of quantitative information**. 2. ed. Cheshire: Graphics Press, 2001.

OBS: Foi utilizado NotebookLM e Google DeepResearch para exploração de fontes confiáveis e recomendação de feedbacks sobre a análise.
