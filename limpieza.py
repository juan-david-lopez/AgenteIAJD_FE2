import pandas as pd
import os
import kagglehub

# ---------------------------
# 📁 RUTAS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "datasetLimpio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 📥 CARGAR DATASET 1000 (LOCAL)
# ---------------------------
df_1000 = pd.read_csv("imdb_top_1000.csv")

# ---------------------------
# 📥 DESCARGAR DATASET 5000 (KAGGLEHUB)
# ---------------------------
path = kagglehub.dataset_download("rakkesharv/imdb-5000-movies-multiple-genres-dataset")

# Buscar automáticamente el CSV dentro del dataset
files = os.listdir(path)
csv_file = [f for f in files if f.endswith(".csv")][0]

csv_path = os.path.join(path, csv_file)
print(f"Usando archivo: {csv_path}")

df_5000 = pd.read_csv(csv_path)

# ---------------------------
# 🔧 LIMPIEZA DATASET 1000
# ---------------------------
df_1000 = df_1000.rename(columns={
    "Series_Title": "titulo",
    "Released_Year": "anio",
    "Runtime": "duracion_raw",
    "Genre": "genero",
    "IMDB_Rating": "calificacion"
})

df_1000["duracion_min"] = df_1000["duracion_raw"].str.replace(" min", "").astype(int)

# ---------------------------
# 🔧 LIMPIEZA DATASET 5000
# ---------------------------
# ⚠️ Ajuste robusto según columnas reales
columnas = df_5000.columns
print("Columnas dataset 5000:", columnas)

# Rename dinámico (según dataset real)
rename_map = {}

if "Movie_Title" in columnas:
    rename_map["Movie_Title"] = "titulo"
if "Year" in columnas:
    rename_map["Year"] = "anio"
if "Runtime(Mins)" in columnas:
    rename_map["Runtime(Mins)"] = "duracion_min"
if "Rating" in columnas:
    rename_map["Rating"] = "calificacion"
if "main_genre" in columnas:
    rename_map["main_genre"] = "genero"

df_5000 = df_5000.rename(columns=rename_map)

# Limpiar nulos (solo si columnas existen)
cols_needed = ["titulo", "duracion_min", "genero"]
cols_exist = [c for c in cols_needed if c in df_5000.columns]

df_5000 = df_5000.dropna(subset=cols_exist)

# ---------------------------
# 🎯 FUNCIONES
# ---------------------------
def clasificar_duracion(minutos):
    try:
        minutos = int(minutos)
        if minutos < 90:
            return "corta"
        elif minutos <= 120:
            return "media"
        else:
            return "larga"
    except:
        return "media"

def inferir_tono(genero):
    genero = str(genero).lower()
    if "comedy" in genero:
        return "divertido"
    elif "drama" in genero:
        return "serio"
    elif "action" in genero:
        return "emocionante"
    elif "horror" in genero:
        return "oscuro"
    elif "biography" in genero:
        return "inspirador"
    else:
        return "neutral"

# ---------------------------
# 🧠 TRANSFORMACIONES
# ---------------------------
df_1000["duracion"] = df_1000["duracion_min"].apply(clasificar_duracion)
df_5000["duracion"] = df_5000["duracion_min"].apply(clasificar_duracion)

df_1000["tono"] = df_1000["genero"].apply(inferir_tono)
df_5000["tono"] = df_5000["genero"].apply(inferir_tono)

# ---------------------------
# 📊 SELECCIÓN FINAL (segura)
# ---------------------------
cols_final = ["titulo", "genero", "duracion", "tono", "anio", "calificacion"]

df_1000 = df_1000[cols_final]
df_5000 = df_5000[cols_final]

# ---------------------------
# 🔥 MERGE
# ---------------------------
df_total = pd.concat([df_1000, df_5000], ignore_index=True)

# Limpiar duplicados por título (normalizado)
df_total["titulo"] = df_total["titulo"].str.lower().str.strip()
df_total = df_total.drop_duplicates(subset=["titulo"])

df_total = df_total.reset_index(drop=True)

# ---------------------------
# 💾 GUARDAR
# ---------------------------
output_path = os.path.join(OUTPUT_DIR, "peliculas_total_limpio.csv")
df_total.to_csv(output_path, index=False)

print(f"\nDataset final generado en: {output_path}")
print(f"Total de películas: {len(df_total)}")