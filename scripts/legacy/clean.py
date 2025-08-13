import pandas as pd

print("📚 Cargando dataset original...")
df_original = pd.read_csv('tracks_features.csv', 
                         sep=',',  # separador original
                         encoding='utf-8',
                         on_bad_lines='skip')

print(f"✅ Dataset cargado: {len(df_original):,} filas")

# Guardar dataset completo saneado
print("💾 Guardando dataset completo saneado...")
df_original.to_csv('tracks_features_clean.csv', 
                   index=False, 
                   sep=';',      # nuevo separador de campos
                   decimal=',')  # nuevo separador decimal

# Guardar muestra de 500 registros saneados
print("💾 Guardando muestra de 500 saneados...")
df_sample = df_original.head(500)
df_sample.to_csv('tracks_features_500.csv', 
                 index=False, 
                 sep=';', 
                 decimal=',')

print("✅ Archivos guardados:")
print("  - tracks_features_clean.csv (completo saneado)")
print("  - tracks_features_500.csv (500 registros saneados)")