#!/usr/bin/env python3
"""
Análisis Exploratorio Completo del Dataset de Canciones con Letras

Este script ejecuta el análisis exploratorio completo del dataset picked_data_lyrics.csv
que contiene 9,987 canciones con letras y características musicales de Spotify.

Uso:
    python exploratory_analysis/run_full_analysis.py
    
Salidas:
    - Reportes comprensivos en outputs/reports/
    - Visualizaciones en formato PNG
    - Análisis estadístico detallado
    - Recomendaciones para clustering
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from exploratory_analysis.reporting.report_generator import ReportGenerator
from exploratory_analysis.data_loading.data_loader import DataLoader
from exploratory_analysis.statistical_analysis.descriptive_stats import DescriptiveStats
from exploratory_analysis.visualization.distribution_plots import DistributionPlotter
from exploratory_analysis.visualization.correlation_heatmaps import CorrelationPlotter
from exploratory_analysis.feature_analysis.dimensionality_reduction import DimensionalityReducer

def print_header():
    """Imprime el encabezado del análisis"""
    print("🎵" + "="*78 + "🎵")
    print("           ANÁLISIS EXPLORATORIO COMPLETO - DATASET MUSICAL")
    print("🎵" + "="*78 + "🎵")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📁 Dataset: data/final_data/picked_data_lyrics.csv")
    print("🎯 Objetivo: Análisis exploratorio para clustering musical")
    print("-" * 80)

def load_and_validate_data():
    """Carga y valida el dataset completo"""
    print("\n📊 PASO 1: CARGA Y VALIDACIÓN DE DATOS")
    print("-" * 50)
    
    loader = DataLoader()
    print("🔄 Cargando dataset completo picked_data_lyrics.csv...")
    
    start_time = time.time()
    result = loader.load_dataset('lyrics_dataset', sample_size=None, validate=True)
    load_time = time.time() - start_time
    
    if not result.success:
        print("❌ Error al cargar el dataset:")
        for error in result.errors:
            print(f"   - {error}")
        return None, None
    
    data = result.data
    print(f"✅ Dataset cargado exitosamente en {load_time:.2f} segundos")
    print(f"📏 Dimensiones: {data.shape[0]:,} filas × {data.shape[1]} columnas")
    print(f"💾 Memoria utilizada: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Mostrar información básica del dataset
    print(f"\n📋 Información del Dataset:")
    print(f"   • Canciones totales: {len(data):,}")
    print(f"   • Características: {len(data.columns)}")
    print(f"   • Datos faltantes: {data.isnull().sum().sum():,}")
    print(f"   • Duplicados: {data.duplicated().sum():,}")
    
    # Mostrar características musicales disponibles
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"   • Características numéricas: {len(numeric_cols)}")
    print(f"     {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")
    
    return data, load_time

def perform_statistical_analysis(data):
    """Ejecuta análisis estadístico completo"""
    print("\n📈 PASO 2: ANÁLISIS ESTADÍSTICO DESCRIPTIVO")
    print("-" * 50)
    
    start_time = time.time()
    stats_analyzer = DescriptiveStats()
    
    print("🔄 Calculando estadísticas descriptivas...")
    stats_results = stats_analyzer.analyze_dataset(data)
    analysis_time = time.time() - start_time
    
    print(f"✅ Análisis estadístico completado en {analysis_time:.2f} segundos")
    
    # Mostrar resumen de estadísticas
    if 'feature_stats' in stats_results:
        feature_stats = stats_results['feature_stats']
        print(f"📊 Estadísticas calculadas para {len(feature_stats)} características")
        
        # Mostrar algunas estadísticas destacadas
        print("\n🎯 Características Musicales Destacadas:")
        musical_features = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness']
        for feature in musical_features:
            if feature in feature_stats:
                stats_obj = feature_stats[feature]
                # FeatureStats es una dataclass, acceder a atributos directamente
                mean_val = getattr(stats_obj, 'mean', 0)
                std_val = getattr(stats_obj, 'std', 0)
                print(f"   • {feature.capitalize()}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    # Información de correlaciones
    if 'correlation_preview' in stats_results:
        corr_info = stats_results['correlation_preview']
        # Manejar tanto dict como objeto
        if hasattr(corr_info, 'high_correlations'):
            high_corr = len(corr_info.high_correlations)
        else:
            high_corr = len(corr_info.get('high_correlations', []))
        print(f"\n🔗 Correlaciones altas detectadas: {high_corr}")
    
    return stats_results, analysis_time

def generate_visualizations(data):
    """Genera visualizaciones del dataset"""
    print("\n🎨 PASO 3: GENERACIÓN DE VISUALIZACIONES")
    print("-" * 50)
    
    start_time = time.time()
    
    # Crear directorio de salida
    output_dir = Path("outputs/reports/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Generando distribuciones de características...")
    
    # Generar distribuciones
    dist_plotter = DistributionPlotter()
    musical_features = ['danceability', 'energy', 'valence', 'speechiness', 'acousticness',
                       'instrumentalness', 'liveness', 'tempo', 'loudness']
    
    available_features = [f for f in musical_features if f in data.columns]
    
    if available_features:
        dist_results = dist_plotter.plot_feature_distributions(
            data, 
            features=available_features[:6],  # Primeras 6 características
            plot_types=['histogram', 'boxplot']
        )
        print(f"✅ Distribuciones generadas para {len(available_features)} características")
    
    # Generar mapas de calor de correlación
    print("🔄 Generando mapas de calor de correlación...")
    corr_plotter = CorrelationPlotter()
    
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if len(numeric_data.columns) >= 2:
        corr_fig = corr_plotter.create_correlation_heatmap(numeric_data)
        if corr_fig:
            corr_path = output_dir / "correlation_heatmap.png"
            corr_fig.savefig(corr_path, dpi=300, bbox_inches='tight')
            print(f"✅ Mapa de calor guardado: {corr_path}")
            
            # Cerrar figura para liberar memoria
            import matplotlib.pyplot as plt
            plt.close(corr_fig)
    
    viz_time = time.time() - start_time
    print(f"✅ Visualizaciones completadas en {viz_time:.2f} segundos")
    
    return viz_time

def perform_dimensionality_analysis(data):
    """Ejecuta análisis de dimensionalidad y PCA"""
    print("\n🔬 PASO 4: ANÁLISIS DE DIMENSIONALIDAD (PCA)")
    print("-" * 50)
    
    start_time = time.time()
    
    # Seleccionar solo características numéricas
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    print(f"🔄 Analizando {len(numeric_data.columns)} características numéricas...")
    
    reducer = DimensionalityReducer()
    
    # Análisis PCA
    pca_results = reducer.fit_pca(numeric_data, variance_threshold=0.90)
    
    if pca_results:
        n_components = pca_results.get('n_components', 0)
        variance_explained = pca_results.get('total_variance_explained', 0)
        
        print(f"✅ PCA completado:")
        print(f"   • Componentes requeridos: {n_components}")
        print(f"   • Varianza explicada: {variance_explained:.1%}")
        
        # Mostrar los primeros componentes
        if 'component_analysis' in pca_results:
            comp_analysis = pca_results['component_analysis']
            print(f"\n🎯 Primeros Componentes Principales:")
            for i, (comp_name, comp_info) in enumerate(list(comp_analysis.items())[:3]):
                # Manejar tanto dict como objeto
                if hasattr(comp_info, 'explained_variance_ratio'):
                    var_ratio = comp_info.explained_variance_ratio
                    interpretation = getattr(comp_info, 'interpretation', 'N/A')
                else:
                    var_ratio = comp_info.get('explained_variance_ratio', 0)
                    interpretation = comp_info.get('interpretation', 'N/A')
                print(f"   • {comp_name}: {var_ratio:.1%} - {interpretation}")
    else:
        print("⚠️  PCA no pudo completarse - posiblemente insuficientes características numéricas")
    
    # t-SNE para una muestra (computacionalmente intensivo)
    sample_size = min(1000, len(numeric_data))
    if len(numeric_data) > sample_size:
        print(f"🔄 Ejecutando t-SNE en muestra de {sample_size:,} canciones...")
        sample_data = numeric_data.sample(n=sample_size, random_state=42)
        tsne_results = reducer.fit_tsne(sample_data, n_components=2)
        
        if tsne_results:
            # Manejar tanto dict como objeto
            if hasattr(tsne_results, 'kl_divergence'):
                kl_div = tsne_results.kl_divergence
            else:
                kl_div = tsne_results.get('kl_divergence', 0)
            print(f"✅ t-SNE completado (KL divergence: {kl_div:.4f})")
    
    analysis_time = time.time() - start_time
    print(f"✅ Análisis de dimensionalidad completado en {analysis_time:.2f} segundos")
    
    return pca_results, analysis_time

def generate_comprehensive_report(data, stats_results, pca_results, total_time):
    """Genera el reporte comprensivo final"""
    print("\n📋 PASO 5: GENERACIÓN DE REPORTE COMPRENSIVO")
    print("-" * 50)
    
    start_time = time.time()
    
    # Usar el generador de reportes
    report_generator = ReportGenerator("outputs/reports/")
    
    print("🔄 Generando reporte comprensivo...")
    report_paths = report_generator.generate_comprehensive_report(
        dataset_type='lyrics_dataset',
        sample_size=None,  # Dataset completo
        include_visualizations=True,
        formats=['json', 'markdown', 'html']
    )
    
    report_time = time.time() - start_time
    
    print(f"✅ Reportes generados en {report_time:.2f} segundos")
    print("\n📁 Archivos de salida:")
    for format_type, path in report_paths.items():
        if path:
            print(f"   • {format_type.upper()}: {path}")
    
    return report_paths, report_time

def print_summary(data, total_time, load_time, analysis_time, viz_time, pca_time, report_time):
    """Imprime resumen final del análisis"""
    print("\n🎉 ANÁLISIS EXPLORATORIO COMPLETADO")
    print("🎵" + "="*78 + "🎵")
    
    print(f"\n📊 RESUMEN DEL ANÁLISIS:")
    print(f"   • Dataset procesado: {len(data):,} canciones")
    print(f"   • Características analizadas: {len(data.select_dtypes(include=['float64', 'int64']).columns)}")
    print(f"   • Tiempo total: {total_time:.2f} segundos")
    
    print(f"\n⏱️  DESGLOSE DE TIEMPOS:")
    print(f"   • Carga de datos: {load_time:.2f}s")
    print(f"   • Análisis estadístico: {analysis_time:.2f}s")
    print(f"   • Visualizaciones: {viz_time:.2f}s") 
    print(f"   • Análisis PCA: {pca_time:.2f}s")
    print(f"   • Generación reportes: {report_time:.2f}s")
    
    print(f"\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
    print("   1. Revisar los reportes generados en outputs/reports/")
    print("   2. Analizar las correlaciones entre características")
    print("   3. Evaluar los resultados del PCA para clustering")
    print("   4. Proceder con el análisis de clustering K-means")
    
    print("\n✅ El dataset está listo para clustering musical!")
    print("🎵" + "="*78 + "🎵")

def main():
    """Función principal que ejecuta todo el análisis"""
    overall_start = time.time()
    
    print_header()
    
    try:
        # Paso 1: Cargar datos
        data, load_time = load_and_validate_data()
        if data is None:
            print("❌ No se pudo cargar el dataset. Abortando análisis.")
            return 1
        
        # Paso 2: Análisis estadístico
        stats_results, analysis_time = perform_statistical_analysis(data)
        
        # Paso 3: Visualizaciones
        viz_time = generate_visualizations(data)
        
        # Paso 4: Análisis de dimensionalidad
        pca_results, pca_time = perform_dimensionality_analysis(data)
        
        # Paso 5: Reporte final
        report_paths, report_time = generate_comprehensive_report(
            data, stats_results, pca_results, time.time() - overall_start
        )
        
        # Resumen final
        total_time = time.time() - overall_start
        print_summary(data, total_time, load_time, analysis_time, viz_time, pca_time, report_time)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ANÁLISIS:")
        print(f"   {str(e)}")
        print("\n🔧 Verifica que:")
        print("   • El archivo picked_data_lyrics.csv existe")
        print("   • Las dependencias están instaladas")
        print("   • Hay suficiente memoria disponible")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)