# 1_cleaned - Dataset Limpio para Testing

## Archivos

| Archivo | Registros | Uso |
|---------|-----------|-----|
| tracks_features_clean.csv | 1,204,025 | Referencia (no usar) |
| tracks_features_500.csv | 500 | Testing rapido |

## Especificaciones

| Atributo | Valor |
|----------|-------|
| Separador | `;` |
| Encoding | UTF-8 |
| Tiene letras | NO |

## Carga

```python
import pandas as pd
# Para testing rapido (500 registros)
df = pd.read_csv('data/1_cleaned/tracks_features_500.csv', sep=';', encoding='utf-8')
```

## Uso

**SOLO PARA DESARROLLO Y DEBUGGING**

- `tracks_features_500.csv`: Muestra pequena para tests unitarios y debugging
- `tracks_features_clean.csv`: Dataset completo limpiado, solo referencia

## Limitaciones

- Sin letras de canciones
- Sin informacion de generos
- No apto para analisis multimodal
- No apto para clustering de produccion

## Origen

Derivado de `0_raw/tracks_features.csv` con correccion de formato (separador cambiado de `,` a `;`).
