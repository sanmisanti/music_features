#!/usr/bin/env python3
"""
Script para generar gráficos de distribuciones antes y después de normalización StandardScaler

Este script crea visualizaciones comparativas que muestran cómo la normalización
preserva la forma de las distribuciones mientras las estandariza a media=0, std=1.

Uso:
    python generate_normalization_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")

def load_dataset():
    """Cargar el dataset final unificado"""
    dataset_path = Path("data/final_data/picked_data_optimal.csv")

    if not dataset_path.exists():
        # Buscar alternativos
        alternatives = [
            "data/with_lyrics/spotify_songs_fixed.csv",
            "clustering_evaluation_project/phase1_dataset_unification/aligned_songs_multimodal_20250822_011617.csv"
        ]

        for alt_path in alternatives:
            if Path(alt_path).exists():
                print(f"Usando dataset alternativo: {alt_path}")
                if "spotify_songs_fixed" in alt_path:
                    return pd.read_csv(alt_path, sep='@@', encoding='utf-8')
                else:
                    return pd.read_csv(alt_path, sep='^', encoding='utf-8')

        raise FileNotFoundError("No se encontró ningún dataset válido")

    # Dataset principal
    print(f"Cargando dataset principal: {dataset_path}")
    return pd.read_csv(dataset_path, sep='^', encoding='utf-8')

def get_musical_features(df):
    """Extraer las 12 características musicales Spotify"""
    musical_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]

    available_features = [col for col in musical_features if col in df.columns]
    print(f"Características musicales encontradas: {len(available_features)}/12")
    print(f"Características disponibles: {available_features}")

    return df[available_features].copy()

def create_before_after_plots(original_data, normalized_data, feature_names):
    """Crear gráficos comparativos antes/después de normalización"""

    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    # Crear figura grande
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Distribuciones de Características Musicales: Antes vs Después de Normalización StandardScaler',
                 fontsize=16, fontweight='bold', y=0.98)

    # Aplanar axes para facilitar indexing
    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, feature in enumerate(feature_names):
        ax = axes_flat[i]

        # Datos originales
        ax.hist(original_data[feature], bins=30, alpha=0.6, color='skyblue',
                label='Original', density=True, edgecolor='black', linewidth=0.5)

        # Datos normalizados
        ax.hist(normalized_data[feature], bins=30, alpha=0.6, color='lightcoral',
                label='Normalizado', density=True, edgecolor='black', linewidth=0.5)

        # Configuración del subplot
        ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Añadir estadísticas
        orig_mean = original_data[feature].mean()
        orig_std = original_data[feature].std()
        norm_mean = normalized_data[feature].mean()
        norm_std = normalized_data[feature].std()

        stats_text = f'Original: μ={orig_mean:.2f}, σ={orig_std:.2f}\nNormalizado: μ={norm_mean:.3f}, σ={norm_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ocultar subplots vacíos
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig

def create_summary_statistics_plot(original_data, normalized_data, feature_names):
    """Crear gráfico resumen de estadísticas"""

    # Calcular estadísticas
    orig_stats = original_data.describe()
    norm_stats = normalized_data.describe()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Estadísticas Comparativas: Normalización StandardScaler',
                 fontsize=16, fontweight='bold')

    # 1. Medias
    x_pos = np.arange(len(feature_names))
    ax1.bar(x_pos - 0.2, orig_stats.loc['mean'], 0.4, label='Original', alpha=0.7, color='skyblue')
    ax1.bar(x_pos + 0.2, norm_stats.loc['mean'], 0.4, label='Normalizado', alpha=0.7, color='lightcoral')
    ax1.set_title('Medias (μ)', fontweight='bold')
    ax1.set_ylabel('Valor')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f.replace('_', '\n') for f in feature_names], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 2. Desviaciones estándar
    ax2.bar(x_pos - 0.2, orig_stats.loc['std'], 0.4, label='Original', alpha=0.7, color='skyblue')
    ax2.bar(x_pos + 0.2, norm_stats.loc['std'], 0.4, label='Normalizado', alpha=0.7, color='lightcoral')
    ax2.set_title('Desviaciones Estándar (σ)', fontweight='bold')
    ax2.set_ylabel('Valor')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f.replace('_', '\n') for f in feature_names], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)

    # 3. Rangos (max - min)
    orig_ranges = orig_stats.loc['max'] - orig_stats.loc['min']
    norm_ranges = norm_stats.loc['max'] - norm_stats.loc['min']
    ax3.bar(x_pos - 0.2, orig_ranges, 0.4, label='Original', alpha=0.7, color='skyblue')
    ax3.bar(x_pos + 0.2, norm_ranges, 0.4, label='Normalizado', alpha=0.7, color='lightcoral')
    ax3.set_title('Rangos (Max - Min)', fontweight='bold')
    ax3.set_ylabel('Valor')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f.replace('_', '\n') for f in feature_names], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Coeficientes de variación
    orig_cv = orig_stats.loc['std'] / orig_stats.loc['mean']
    norm_cv = norm_stats.loc['std'] / norm_stats.loc['mean']
    ax4.bar(x_pos - 0.2, orig_cv, 0.4, label='Original', alpha=0.7, color='skyblue')
    ax4.bar(x_pos + 0.2, norm_cv, 0.4, label='Normalizado', alpha=0.7, color='lightcoral')
    ax4.set_title('Coeficientes de Variación (σ/μ)', fontweight='bold')
    ax4.set_ylabel('Ratio')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f.replace('_', '\n') for f in feature_names], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_correlation_comparison_plot(original_data, normalized_data):
    """Crear gráfico comparativo de matrices de correlación"""

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Preservación de Correlaciones tras Normalización StandardScaler',
                 fontsize=16, fontweight='bold')

    # Matriz de correlación original
    corr_orig = original_data.corr()
    sns.heatmap(corr_orig, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlaciones Originales', fontweight='bold')

    # Matriz de correlación normalizada
    corr_norm = normalized_data.corr()
    sns.heatmap(corr_norm, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Correlaciones Normalizadas', fontweight='bold')

    # Diferencia (debe ser prácticamente cero)
    corr_diff = np.abs(corr_orig - corr_norm)
    sns.heatmap(corr_diff, annot=True, cmap='Reds',
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Diferencias Absolutas\n(Verificación de Preservación)', fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    """Función principal"""
    print("🎵 Generando gráficos de distribuciones antes/después de normalización StandardScaler")
    print("=" * 80)

    # Cargar datos
    print("📂 Cargando dataset...")
    df = load_dataset()
    print(f"✅ Dataset cargado: {df.shape[0]:,} canciones, {df.shape[1]} columnas")

    # Extraer características musicales
    print("\n🎼 Extrayendo características musicales...")
    musical_data = get_musical_features(df)
    print(f"✅ Características extraídas: {musical_data.shape}")

    # Aplicar normalización
    print("\n📊 Aplicando normalización StandardScaler...")
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(musical_data),
        columns=musical_data.columns,
        index=musical_data.index
    )
    print("✅ Normalización aplicada")

    # Verificar normalización
    print(f"\n📈 Verificación de normalización:")
    print(f"  Original - Media: {musical_data.mean().mean():.3f}, Std: {musical_data.std().mean():.3f}")
    print(f"  Normalizado - Media: {normalized_data.mean().mean():.6f}, Std: {normalized_data.std().mean():.6f}")

    # Crear directorio de salida
    output_dir = Path("INFORME_FINAL/imagenes_normalizacion")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n📁 Directorio de salida: {output_dir}")

    # Generar gráficos
    print("\n🎨 Generando gráficos...")

    # 1. Distribuciones antes/después
    print("  1️⃣ Distribuciones comparativas...")
    fig1 = create_before_after_plots(musical_data, normalized_data, musical_data.columns)
    fig1.savefig(output_dir / "distribuciones_antes_despues_normalizacion.png",
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # 2. Estadísticas resumen
    print("  2️⃣ Estadísticas comparativas...")
    fig2 = create_summary_statistics_plot(musical_data, normalized_data, musical_data.columns)
    fig2.savefig(output_dir / "estadisticas_comparativas_normalizacion.png",
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    # 3. Correlaciones
    print("  3️⃣ Preservación de correlaciones...")
    fig3 = create_correlation_comparison_plot(musical_data, normalized_data)
    fig3.savefig(output_dir / "correlaciones_preservacion_normalizacion.png",
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    print("\n✅ ¡Gráficos generados exitosamente!")
    print(f"📂 Ubicación: {output_dir.absolute()}")
    print("\n📋 Archivos creados:")
    for img_file in output_dir.glob("*.png"):
        print(f"  - {img_file.name}")

    print("\n🎯 Los gráficos muestran:")
    print("  ✓ Preservación de la forma de las distribuciones")
    print("  ✓ Normalización a media=0, desviación estándar=1")
    print("  ✓ Conservación exacta de correlaciones entre variables")
    print("  ✓ Eliminación de efectos de escala entre características")

if __name__ == "__main__":
    main()