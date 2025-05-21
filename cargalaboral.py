# -*- coding: utf-8 -*-
"""
Plan de Análisis en Python para Tesis: Evolución del Acceso Odontológico
Período: 2019-2023 | Hogares con madres jefas en Perú
"""

import pandas as pd
import statsmodels.api as sm

# Try to import matplotlib for charts, skip visuals if unavailable
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Panel data models
# pip install linearmodels
from linearmodels.panel import PanelOLS, RandomEffects

# --------------------------------------------------
# 1. Load and select relevant variables
# --------------------------------------------------
# Load from Google Sheets as CSV
sheet_id = "19ifjmMaZQceVro3Hew-XdIgCxXvO2aGqYk1i_c1BhSM"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
df = pd.read_csv(csv_url)
df.columns = df.columns.str.lower()

# Variables according to operationalization
vars_keep = [
    'p414n_06_19','p414n_06_20','p414n_06_21','p414n_06_22','p414n_06_23',  # access
    'p208a_19','p208a_20','p208a_21','p208a_22','p208a_23',                  # age
    'p209_19','p209_20','p209_21','p209_22','p209_23',                      # civil status
    'p520_19','p520_20','p520_21','p520_22','p520_23',                      # work hours
    'ocupinf_19','ocupinf_20','ocupinf_21','ocupinf_22','ocupinf_23',        # employment type
    'p301a_19','p301a_20','p301a_21','p301a_22','p301a_23',                # education
    'estrato_19','estrato_20','estrato_21','estrato_22','estrato_23',      # residence
    'dominio_19','dominio_20','dominio_21','dominio_22','dominio_23',      # region
    'ingmo2hd_19','ingmo2hd_20','ingmo2hd_21','ingmo2hd_22','ingmo2hd_23',  # income
    'pobreza_19','pobreza_20','pobreza_21','pobreza_22','pobreza_23',      # poverty
    'p4191_19','p4191_20','p4191_21','p4193_22','p4191_23'               # insurance
]
df = df[[v for v in vars_keep if v in df.columns]]
# Create panel ID if not present
df['id'] = df.index

# --------------------------------------------------
# 2. Reshape to long-format panel
# --------------------------------------------------
suffixes = ['19','20','21','22','23']
years    = [2019,2020,2021,2022,2023]
long_list = []
for suf, year in zip(suffixes, years):
    temp = pd.DataFrame({
        'id': df['id'],
        'year': year,
        'access': df[f'p414n_06_{suf}'],
        'age_mom': df[f'p208a_{suf}'],
        'civil': df[f'p209_{suf}'],
        'hours': df[f'p520_{suf}'],
        'emp_type': df[f'ocupinf_{suf}'],
        'edu': df[f'p301a_{suf}'],
        'residence': df[f'estrato_{suf}'],
        'region': df[f'dominio_{suf}'],
        'income': df[f'ingmo2hd_{suf}'],
        'poverty': df[f'pobreza_{suf}'],
        'insurance': df[f'p4193_{suf}'] if suf=='22' else df[f'p4191_{suf}']
    })
    long_list.append(temp)
df_long = pd.concat(long_list, ignore_index=True)

# --------------------------------------------------
# 3. Create derived variables
# --------------------------------------------------
def categorize_hours(h):
    if pd.isna(h): return pd.NA
    if h <= 20: return 0
    if h <= 40: return 1
    if h <= 48: return 2
    return 3

df_long['hours_cat'] = df_long['hours'].apply(categorize_hours)
df_long['access_bin'] = df_long['access'].map({1:1,2:0}).fillna(0).astype(int)
df_long['urban'] = df_long['residence'].apply(lambda x: 1 if x in [1,2,3,4,5] else 0 if x in [6,7,8] else pd.NA)

# --------------------------------------------------
# 4. Descriptive and bivariate analysis
# --------------------------------------------------
print("Frequency of access:")
print(df_long['access_bin'].value_counts(dropna=False))
print(df_long.groupby('year')['access_bin'].mean())
print(df_long.groupby('hours_cat')['access_bin'].mean())
print(df_long.groupby('edu')['access_bin'].mean())
print(df_long.groupby('urban')['access_bin'].mean())

# Correlations
corr_list = ['access_bin','hours','age_mom','income']
print(df_long[corr_list].corr())
print(df_long[corr_list].corr(method='spearman'))

# --------------------------------------------------
# 5. Visualizations (if available)
# --------------------------------------------------
if HAS_MATPLOTLIB:
    df_long.groupby('year')['access_bin'].mean().plot(marker='o')
    plt.title('Access trend')
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.show()

    df_long.groupby('hours_cat')['access_bin'].mean().plot(kind='bar')
    plt.title('Access vs. Hours Category')
    plt.xlabel('Hours Category')
    plt.ylabel('Proportion')
    plt.show()

# --------------------------------------------------
# 6. Panel econometric models and robustness
# --------------------------------------------------
df_panel = df_long.dropna(subset=['access_bin','hours_cat']).set_index(['id','year'])
# Dummies for categorical exogenous vars
dummies = pd.get_dummies(df_panel[['hours_cat','urban','emp_type','edu','region','insurance']], drop_first=True)
exog = sm.add_constant(pd.concat([dummies, df_panel[['age_mom','income','poverty']]], axis=1))

y = df_panel['access_bin']
# Fixed effects
fe_res = PanelOLS(y, exog, entity_effects=True).fit()
print(fe_res.summary)
# Random effects
re_res = RandomEffects(y, exog).fit()
print(re_res.summary)
# Hausman test
from linearmodels.panel import compare
print(compare({'FE': fe_res, 'RE': re_res}))
```
