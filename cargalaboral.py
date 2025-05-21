# -*- coding: utf-8 -*-
"""
Plan de Análisis en Python para Tesis: Evolución del Acceso Odontológico
Período: 2019-2023 | Hogares con madres jefas en Perú
"""

import pandas as pd

# Intento importar statsmodels; si no está disponible se omite modelado
try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("Warning: statsmodels no está instalado; se omitirá el modelado estadístico.")

# Intento importar matplotlib; si no está disponible se omiten visualizaciones
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib no está instalado; se omitirán figuras.")

# Intento importar modelos de panel
try:
    from linearmodels.panel import PanelOLS, RandomEffects, compare
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False
    print("Warning: linearmodels no está instalado; se omitirá el modelado de panel.")

# --------------------------------------------------
# 1. Load and select relevant variables
# --------------------------------------------------
# Load from Google Sheets as CSV
sheet_id = "19ifjmMaZQceVro3Hew-XdIgCxXvO2aGqYk1i_c1BhSM"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
df = pd.read_csv(csv_url)
df.columns = df.columns.str.lower()

# Variables según operacionalización
to_keep = [
    'p414n_06_19','p414n_06_20','p414n_06_21','p414n_06_22','p414n_06_23',
    'p208a_19','p208a_20','p208a_21','p208a_22','p208a_23',
    'p209_19','p209_20','p209_21','p209_22','p209_23',
    'p520_19','p520_20','p520_21','p520_22','p520_23',
    'ocupinf_19','ocupinf_20','ocupinf_21','ocupinf_22','ocupinf_23',
    'p301a_19','p301a_20','p301a_21','p301a_22','p301a_23',
    'estrato_19','estrato_20','estrato_21','estrato_22','estrato_23',
    'dominio_19','dominio_20','dominio_21','dominio_22','dominio_23',
    'ingmo2hd_19','ingmo2hd_20','ingmo2hd_21','ingmo2hd_22','ingmo2hd_23',
    'pobreza_19','pobreza_20','pobreza_21','pobreza_22','pobreza_23',
    'p4191_19','p4191_20','p4191_21','p4193_22','p4191_23'
]
df = df[[v for v in to_keep if v in df.columns]]
# Crear ID de panel si no existe
if 'id' not in df.columns:
    df['id'] = df.index

# --------------------------------------------------
# 2. Transform to long-format panel
# --------------------------------------------------
suffixes = ['19','20','21','22','23']
years = [2019,2020,2021,2022,2023]
frames = []
for suf, yr in zip(suffixes, years):
    temp = pd.DataFrame({
        'id': df['id'],
        'year': yr,
        'access': df.get(f'p414n_06_{suf}'),
        'age_mom': df.get(f'p208a_{suf}'),
        'civil': df.get(f'p209_{suf}'),
        'hours': df.get(f'p520_{suf}'),
        'emp_type': df.get(f'ocupinf_{suf}'),
        'edu': df.get(f'p301a_{suf}'),
        'residence': df.get(f'estrato_{suf}'),
        'region': df.get(f'dominio_{suf}'),
        'income': df.get(f'ingmo2hd_{suf}'),
        'poverty': df.get(f'pobreza_{suf}'),
        'insurance': df.get(f'p4193_{suf}') if suf=='22' else df.get(f'p4191_{suf}')
    })
    frames.append(temp)
import itertools
# Concatenate
df_long = pd.concat(frames, ignore_index=True)

# --------------------------------------------------
# 3. Derived variables
# --------------------------------------------------
def categorize_hours(x):
    if pd.isna(x): return pd.NA
    return 0 if x<=20 else 1 if x<=40 else 2 if x<=48 else 3
df_long['hours_cat'] = df_long['hours'].apply(categorize_hours)
df_long['access_bin'] = df_long['access'].map({1:1,2:0}).fillna(0).astype(int)
df_long['urban'] = df_long['residence'].apply(lambda v: 1 if v in [1,2,3,4,5] else 0 if v in [6,7,8] else pd.NA)

# --------------------------------------------------
# 4. Descriptive & bivariate
# --------------------------------------------------
print(df_long['access_bin'].value_counts(dropna=False))
print(df_long.groupby('year')['access_bin'].mean())

# Correlations if statsmodels available
if HAS_SM:
    corr_vars = ['access_bin','hours','age_mom','income']
    print(df_long[corr_vars].corr())
    print(df_long[corr_vars].corr(method='spearman'))

# --------------------------------------------------
# 5. Visuals
# --------------------------------------------------
if HAS_MATPLOTLIB:
    df_long.groupby('year')['access_bin'].mean().plot(marker='o'); plt.show()
    df_long.groupby('hours_cat')['access_bin'].mean().plot(kind='bar'); plt.show()

# --------------------------------------------------
# --------------------------------------------------
# 6. Panel modeling (requires statsmodels and linearmodels)
# --------------------------------------------------
if HAS_PANEL and HAS_SM:
    # Re-import statsmodels inside guarded block to ensure availability
    import statsmodels.api as sm
    from linearmodels.panel import compare

    # Prepare panel data
    df_panel = df_long.dropna(subset=['access_bin','hours_cat']).set_index(['id','year'])
    dummies = pd.get_dummies(
        df_panel[['hours_cat','urban','emp_type','edu','region','insurance']],
        drop_first=True
    )
    exog = sm.add_constant(
        pd.concat([dummies, df_panel[['age_mom','income','poverty']]], axis=1)
    )
    y = df_panel['access_bin']

    # Fixed effects model
    fe = PanelOLS(y, exog, entity_effects=True).fit()
    print("
Fixed Effects Model Summary:")
    print(fe.summary)

    # Random effects model
    re = RandomEffects(y, exog).fit()
    print("
Random Effects Model Summary:")
    print(re.summary)

    # Hausman test
    hausman = compare({'FE': fe, 'RE': re})
    print("
Hausman Test Result:")
    print(hausman)
else:
    if not HAS_PANEL:
        print("Warning: linearmodels is not available; panel modeling skipped.")
    if not HAS_SM:
        print("Warning: statsmodels is not available; panel modeling skipped.")
