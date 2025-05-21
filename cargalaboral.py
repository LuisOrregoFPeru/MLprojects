plan de análisis en Python basado en tus objetivos, hipótesis y la tabla de operacionalización, organizado en bloques claros y reproducibles, siguiendo buenas prácticas científicas. El código incluye:

Estructuración de datos de panel,

Análisis descriptivo,

Análisis bivariado (correlaciones y pruebas de hipótesis),

Modelado econométrico de panel (efectos fijos y aleatorios),

Pruebas estadísticas de robustez.

He documentado cada bloque, para que puedas copiar, adaptar y ejecutar por partes según tus necesidades y recursos.

1. Carga y selección de variables relevantes
python
Copy
Edit
import pandas as pd

# 1. Carga y selección de variables relevantes
# --------------------------------------------------
# Ruta al Google Sheet (exportación CSV)
# Asegúrate de que la hoja deseada sea la primera (gid=0) o ajusta el parámetro gid
sheet_id = "19ifjmMaZQceVro3Hew-XdIgCxXvO2aGqYk1i_c1BhSM"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"

df = pd.read_csv(csv_url)
# Si tuvieras un .dta local, podrías usar:
# df = pd.read_stata(ruta_dta)

df.columns = df.columns.str.lower()

# 2. Selección de variables relevantes (según operacionalización)
vars_keep = [
    # Acceso odontológico
    'p414n_06_19', 'p414n_06_20', 'p414n_06_21', 'p414n_06_22', 'p414n_06_23',
    # Año de panel (identificador)
    'hpanel_19_20', 'hpanel_20_21', 'hpanel_21_22', 'hpanel_22_23',
    # Edad madre jefa hogar
    'p208a_19', 'p208a_20', 'p208a_21', 'p208a_22', 'p208a_23',
    # Estado civil madre jefa hogar
    'p209_19', 'p209_20', 'p209_21', 'p209_22', 'p209_23',
    # Carga laboral (horas trabajadas por semana)
    'p520_19', 'p520_20', 'p520_21', 'p520_22', 'p520_23',
    # Tipo de empleo (formal/informal)
    'ocupinf_19', 'ocupinf_20', 'ocupinf_21', 'ocupinf_22', 'ocupinf_23',
    # Nivel educativo madre jefa hogar
    'p301a_19', 'p301a_20', 'p301a_21', 'p301a_22', 'p301a_23',
    # Área de residencia
    'estrato_19', 'estrato_20', 'estrato_21', 'estrato_22', 'estrato_23',
    # Región geográfica
    'dominio_19', 'dominio_20', 'dominio_21', 'dominio_22', 'dominio_23',
    # Nivel de ingresos
    'ingmo2hd_19', 'ingmo2hd_20', 'ingmo2hd_21', 'ingmo2hd_22', 'ingmo2hd_23',
    # Quintil de pobreza
    'pobreza_19', 'pobreza_20', 'pobreza_21', 'pobreza_22', 'pobreza_23',
    # Seguro de salud
    'p4191_19', 'p4191_20', 'p4191_21', 'p4193_22', 'p4191_23'
]
df = df[[v for v in vars_keep if v in df.columns]]
2. Transformación a formato panel largo (“long format”)
python
Copy
Edit
# Definir los años de análisis
anios = [2019, 2020, 2021, 2022, 2023]

# Diccionario para el reshape
reshape_dict = {
    'acceso_odontologico': ['p414n_06_19', 'p414n_06_20', 'p414n_06_21', 'p414n_06_22', 'p414n_06_23'],
    'edad_madre': ['p208a_19', 'p208a_20', 'p208a_21', 'p208a_22', 'p208a_23'],
    'estado_civil': ['p209_19', 'p209_20', 'p209_21', 'p209_22', 'p209_23'],
    'carga_laboral': ['p520_19', 'p520_20', 'p520_21', 'p520_22', 'p520_23'],
    'tipo_empleo': ['ocupinf_19', 'ocupinf_20', 'ocupinf_21', 'ocupinf_22', 'ocupinf_23'],
    'nivel_educativo': ['p301a_19', 'p301a_20', 'p301a_21', 'p301a_22', 'p301a_23'],
    'estrato': ['estrato_19', 'estrato_20', 'estrato_21', 'estrato_22', 'estrato_23'],
    'region': ['dominio_19', 'dominio_20', 'dominio_21', 'dominio_22', 'dominio_23'],
    'ingresos': ['ingmo2hd_19', 'ingmo2hd_20', 'ingmo2hd_21', 'ingmo2hd_22', 'ingmo2hd_23'],
    'pobreza': ['pobreza_19', 'pobreza_20', 'pobreza_21', 'pobreza_22', 'pobreza_23'],
    'seguro': ['p4191_19', 'p4191_20', 'p4191_21', 'p4193_22', 'p4191_23'],
}

# Construir el dataframe en formato largo
df_long = pd.DataFrame()
for idx, anio in enumerate(anios):
    temp = pd.DataFrame({
        'anio': anio,
        'acceso_odontologico': df[reshape_dict['acceso_odontologico'][idx]],
        'edad_madre': df[reshape_dict['edad_madre'][idx]],
        'estado_civil': df[reshape_dict['estado_civil'][idx]],
        'carga_laboral': df[reshape_dict['carga_laboral'][idx]],
        'tipo_empleo': df[reshape_dict['tipo_empleo'][idx]],
        'nivel_educativo': df[reshape_dict['nivel_educativo'][idx]],
        'estrato': df[reshape_dict['estrato'][idx]],
        'region': df[reshape_dict['region'][idx]],
        'ingresos': df[reshape_dict['ingresos'][idx]],
        'pobreza': df[reshape_dict['pobreza'][idx]],
        'seguro': df[reshape_dict['seguro'][idx]],
    })
    df_long = pd.concat([df_long, temp], ignore_index=True)
3. Creación de variables categóricas y de análisis
python
Copy
Edit
# Categorización de carga laboral
def categorizar_carga(horas):
    if pd.isna(horas):
        return None
    if horas <= 20:
        return 0  # tiempo corto
    elif horas <= 40:
        return 1  # tiempo completo
    elif horas <= 48:
        return 2  # tiempo extendido
    else:
        return 3  # muy extendido

df_long['carga_laboral_cat'] = df_long['carga_laboral'].apply(categorizar_carga)

# Categorización de acceso odontológico (puede requerir ajuste según codificación original)
df_long['acceso_odontologico_bin'] = df_long['acceso_odontologico'].map({1: 1, 2: 0, 0: 0})  # Ajusta según tu diccionario

# Recodificación de variables relevantes (puedes extender según operacionalización)
# Ejemplo: área de residencia
df_long['urbano'] = df_long['estrato'].apply(lambda x: 1 if x in [1,2,3,4,5] else 0 if x in [6,7,8] else None)
4. Análisis descriptivo (tendencias y comparación entre grupos)
python
Copy
Edit
# Tablas de frecuencia y estadísticos descriptivos
print(df_long['acceso_odontologico_bin'].value_counts(dropna=False))
print(df_long.groupby('anio')['acceso_odontologico_bin'].mean())

# Distribuciones según características
print(df_long.groupby('carga_laboral_cat')['acceso_odontologico_bin'].mean())
print(df_long.groupby('nivel_educativo')['acceso_odontologico_bin'].mean())
print(df_long.groupby('urbano')['acceso_odontologico_bin'].mean())
5. Visualización de tendencias (gráficos temporales y por grupo)
python
Copy
Edit
import matplotlib.pyplot as plt

# Evolución del acceso odontológico por año
df_long.groupby('anio')['acceso_odontologico_bin'].mean().plot(marker='o')
plt.ylabel("Proporción con acceso odontológico")
plt.xlabel("Año")
plt.title("Evolución del acceso a servicios odontológicos (2019-2023)")
plt.show()

# Acceso según carga laboral
df_long.groupby('carga_laboral_cat')['acceso_odontologico_bin'].mean().plot(kind='bar')
plt.ylabel("Proporción con acceso odontológico")
plt.xlabel("Categoría de carga laboral")
plt.title("Acceso a servicios odontológicos según carga laboral de la madre")
plt.show()
6. Análisis de correlación y colinealidad
python
Copy
Edit
# Correlación de Pearson y Spearman entre las principales variables numéricas
print(df_long[['acceso_odontologico_bin', 'carga_laboral', 'edad_madre', 'ingresos']].corr())
print(df_long[['acceso_odontologico_bin', 'carga_laboral', 'edad_madre', 'ingresos']].corr(method='spearman'))
7. Modelo de regresión de datos de panel
Este bloque requiere las librerías linearmodels o statsmodels para modelos de panel. Primero debes instalar linearmodels si aún no lo has hecho:

bash
Copy
Edit
pip install linearmodels
Luego, ejecuta:

python
Copy
Edit
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# Preparar el DataFrame: necesitas un identificador único por individuo/hogar
# Ejemplo: df_long['id'] = ...  # Asegúrate de tener este identificador (puedes generarlo si no existe)
df_long = df_long.dropna(subset=['acceso_odontologico_bin', 'carga_laboral_cat'])

# Definir variables explicativas
exog_vars = [
    'anio', 'carga_laboral_cat', 'edad_madre', 'nivel_educativo',
    'urbano', 'region', 'ingresos', 'pobreza', 'seguro', 'tipo_empleo'
]
# One-hot encoding para variables categóricas (nivel_educativo, region, seguro, etc.)
df_long = pd.get_dummies(df_long, columns=['nivel_educativo', 'region', 'seguro', 'tipo_empleo'], drop_first=True)

# Definir y ajustar el modelo
y = df_long['acceso_odontologico_bin']
X = sm.add_constant(df_long[exog_vars])
# En modelos de panel: necesitas setear multiindex (id, año)
df_long = df_long.set_index(['id', 'anio'])
mod = PanelOLS(y, X, entity_effects=True)  # Efectos fijos
res = mod.fit()
print(res.summary)
8. Pruebas de Hausman, Breusch-Pagan y Wooldridge (robustez del modelo)
Hausman: Lo hace linearmodels (ver documentación).

Breusch-Pagan: Usa het_breuschpagan de statsmodels.stats.diagnostic.

Wooldridge: Para autocorrelación en panel (ver linearmodels).

