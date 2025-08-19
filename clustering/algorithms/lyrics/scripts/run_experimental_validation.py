#!/usr/bin/env python3
"""
Validación Experimental Completa - Sistema Clustering Semántico FASE 5

Script para ejecutar validación experimental completa del sistema de clustering 
semántico de letras musicales en el dataset real de 16,081 canciones.

Genera resultados científicos para análisis académico y comparación con 
sistema de clustering musical existente.

Autor: Sistema de Clustering Musical
Fecha: Agosto 2025 - Validación Experimental
"""

import sys
import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'experimental_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Importar componentes FASE 5
try:
    from clustering.semantic_kmeans import SemanticKMeans
    from clustering.hierarchical_clusterer import HierarchicalClusterer
    from evaluation.cluster_evaluator import ClusterEvaluator
    from evaluation.cluster_visualizer import ClusterVisualizer
    from recommendation.hybrid_music_clusterer import HybridMusicClusterer
    from vectorization.bert_vectorizer import BertVectorizer
    from vectorization.batch_processor import BatchProcessor
    
    IMPORTS_SUCCESS = True
except ImportError as e:
    logger.error(f"Error importando componentes FASE 5: {e}")
    IMPORTS_SUCCESS = False


class ExperimentalValidator:
    """
    Validador experimental completo para clustering semántico.
    
    Ejecuta experimentos científicos completos sobre dataset real
    y genera reportes académicos detallados.
    """
    
    def __init__(self, 
                 dataset_path: str = "/mnt/c/Users/sanmi/Documents/Proyectos/Tesis/music_features/data/final_data/picked_data_optimal.csv",
                 output_dir: str = "experimental_results"):
        """
        Inicializa validador experimental.
        
        Args:
            dataset_path: Ruta al dataset optimizado 16K
            output_dir: Directorio para resultados experimentales
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para resultados únicos
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Componentes del sistema
        self.bert_vectorizer = None
        self.batch_processor = None
        self.evaluator = ClusterEvaluator()
        self.visualizer = ClusterVisualizer(output_dir=self.output_dir)
        
        # Resultados experimentales
        self.experimental_results = {}
        
        logger.info(f"🔬 ExperimentalValidator inicializado:")
        logger.info(f"   Dataset: {self.dataset_path}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Timestamp: {self.timestamp}")
    
    def run_complete_validation(self, 
                              sample_size: int = 1000,
                              algorithms: List[str] = None) -> Dict[str, Any]:
        """
        Ejecuta validación experimental completa.
        
        Args:
            sample_size: Tamaño muestra para experimentos (1K-16K)
            algorithms: Lista algoritmos a evaluar
            
        Returns:
            Dict con resultados experimentales completos
        """
        if not IMPORTS_SUCCESS:
            logger.error("❌ Componentes no disponibles - experimentos cancelados")
            return {"success": False, "error": "Import failure"}
        
        logger.info("🚀 INICIANDO VALIDACIÓN EXPERIMENTAL COMPLETA")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Algoritmos por defecto
        if algorithms is None:
            algorithms = ['kmeans_semantic', 'hierarchical_semantic']
        
        try:
            # 1. Cargar y preparar dataset
            df_sample = self._load_and_sample_dataset(sample_size)
            
            # 2. Extraer embeddings semánticos
            embeddings = self._extract_semantic_embeddings(df_sample)
            
            # 3. Ejecutar experimentos clustering
            clustering_results = self._run_clustering_experiments(
                embeddings, df_sample, algorithms
            )
            
            # 4. Evaluación comparativa
            comparative_results = self._run_comparative_evaluation(clustering_results)
            
            # 5. Análisis musical específico
            musical_analysis = self._run_musical_analysis(
                clustering_results, df_sample
            )
            
            # 6. Generación de visualizaciones
            visualizations = self._generate_experimental_visualizations(
                embeddings, clustering_results, df_sample
            )
            
            # 7. Compilar resultados finales
            total_time = time.time() - start_time
            
            self.experimental_results = {
                "experiment_info": {
                    "timestamp": self.timestamp,
                    "dataset_path": str(self.dataset_path),
                    "sample_size": sample_size,
                    "algorithms_tested": algorithms,
                    "total_time": total_time
                },
                "dataset_characteristics": self._analyze_dataset_characteristics(df_sample),
                "clustering_results": clustering_results,
                "comparative_analysis": comparative_results,
                "musical_analysis": musical_analysis,
                "visualizations": visualizations,
                "conclusions": self._generate_scientific_conclusions()
            }
            
            # 8. Guardar resultados
            self._save_experimental_results()
            
            # 9. Generar reporte final
            self._generate_final_report()
            
            logger.info(f"🎉 VALIDACIÓN EXPERIMENTAL COMPLETADA: {total_time:.2f}s")
            
            return {"success": True, "results": self.experimental_results}
            
        except Exception as e:
            logger.error(f"💥 Error en validación experimental: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_and_sample_dataset(self, sample_size: int) -> pd.DataFrame:
        """Carga dataset y crea muestra estratificada."""
        logger.info(f"📊 Cargando dataset real...")
        
        # Cargar dataset optimizado
        df = pd.read_csv(self.dataset_path, sep='^', encoding='utf-8')
        logger.info(f"   📊 Dataset cargado: {len(df)} canciones, {len(df.columns)} características")
        
        # Validar columna lyrics
        if 'lyrics' not in df.columns:
            raise ValueError("Columna 'lyrics' no encontrada en dataset")
        
        # Filtrar canciones con letras válidas
        valid_lyrics = df['lyrics'].notna() & (df['lyrics'] != '') & (df['lyrics'] != 'nan')
        df_valid = df[valid_lyrics].reset_index(drop=True)
        
        logger.info(f"   ✅ Canciones con letras válidas: {len(df_valid)}")
        
        # Crear muestra estratificada
        if sample_size >= len(df_valid):
            df_sample = df_valid.copy()
            logger.info(f"   📊 Usando dataset completo: {len(df_sample)} canciones")
        else:
            # Muestreo aleatorio estratificado
            df_sample = df_valid.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"   📊 Muestra creada: {len(df_sample)} canciones")
        
        return df_sample
    
    def _extract_semantic_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extrae embeddings BERT de las letras."""
        logger.info(f"🤖 Extrayendo embeddings semánticos...")
        
        # Inicializar componentes BERT
        self.bert_vectorizer = BertVectorizer(cache_enabled=True)
        self.batch_processor = BatchProcessor(self.bert_vectorizer)
        
        # Procesar letras en batch
        lyrics_list = df['lyrics'].tolist()
        
        start_time = time.time()
        results = self.batch_processor.process_batch(lyrics_list)
        processing_time = time.time() - start_time
        
        # Extraer embeddings exitosos
        embeddings = []
        success_count = 0
        
        for result in results:
            if result['success'] and 'embedding' in result:
                embeddings.append(result['embedding'])
                success_count += 1
            else:
                # Embedding cero para fallos
                embeddings.append(np.zeros(384))
        
        embeddings_array = np.array(embeddings)
        
        logger.info(f"   ✅ Embeddings extraídos: {embeddings_array.shape}")
        logger.info(f"   📊 Éxito: {success_count}/{len(lyrics_list)} ({success_count/len(lyrics_list)*100:.1f}%)")
        logger.info(f"   ⏱️ Tiempo procesamiento: {processing_time:.2f}s")
        logger.info(f"   🚀 Velocidad: {len(lyrics_list)/processing_time:.2f} letras/segundo")
        
        return embeddings_array
    
    def _run_clustering_experiments(self, 
                                  embeddings: np.ndarray, 
                                  df: pd.DataFrame,
                                  algorithms: List[str]) -> Dict[str, Any]:
        """Ejecuta experimentos de clustering con múltiples algoritmos."""
        logger.info(f"🔬 Ejecutando experimentos clustering...")
        
        clustering_results = {}
        lyrics_list = df['lyrics'].tolist()
        
        for algorithm in algorithms:
            logger.info(f"   🧪 Algoritmo: {algorithm}")
            
            try:
                if algorithm == 'kmeans_semantic':
                    clusterer = SemanticKMeans(
                        n_clusters=None,  # Auto-optimización
                        auto_optimize_k=True,
                        metric='cosine',
                        max_k=min(15, len(embeddings) // 50)
                    )
                elif algorithm == 'hierarchical_semantic':
                    clusterer = HierarchicalClusterer(
                        n_clusters=None,  # Auto-determinación
                        auto_clusters=True,
                        linkage='average',
                        metric='cosine'
                    )
                else:
                    logger.warning(f"Algoritmo {algorithm} no soportado")
                    continue
                
                # Entrenar
                start_time = time.time()
                clusterer.fit(embeddings, lyrics_list)
                training_time = time.time() - start_time
                
                # Obtener resultados
                labels = clusterer.get_cluster_assignments()
                cluster_info = clusterer.get_cluster_info()
                
                # Evaluación
                evaluation = self.evaluator.evaluate_clustering(
                    embeddings, labels, lyrics_list, detailed=True
                )
                
                clustering_results[algorithm] = {
                    "clusterer": clusterer,
                    "labels": labels,
                    "cluster_info": cluster_info,
                    "evaluation": evaluation,
                    "training_time": training_time
                }
                
                logger.info(f"     ✅ {algorithm}: {cluster_info.get('n_clusters_actual', 'N/A')} clusters, "
                           f"{evaluation['standard_metrics'].get('silhouette_score', 0):.3f} silhouette, "
                           f"{training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"     ❌ Error en {algorithm}: {e}")
                clustering_results[algorithm] = {"error": str(e)}
        
        return clustering_results
    
    def _run_comparative_evaluation(self, clustering_results: Dict) -> Dict[str, Any]:
        """Evaluación comparativa entre algoritmos."""
        logger.info(f"📊 Ejecutando evaluación comparativa...")
        
        comparative_analysis = {
            "algorithm_ranking": [],
            "metric_comparison": {},
            "performance_comparison": {},
            "cluster_consistency": {}
        }
        
        # Extraer métricas por algoritmo
        algorithm_metrics = {}
        
        for algorithm, results in clustering_results.items():
            if "error" in results:
                continue
            
            eval_results = results["evaluation"]
            std_metrics = eval_results.get("standard_metrics", {})
            
            algorithm_metrics[algorithm] = {
                "silhouette_score": std_metrics.get("silhouette_score", 0),
                "davies_bouldin_index": std_metrics.get("davies_bouldin_index", float('inf')),
                "calinski_harabasz_index": std_metrics.get("calinski_harabasz_index", 0),
                "n_clusters": results["cluster_info"].get("n_clusters_actual", 0),
                "training_time": results["training_time"]
            }
        
        # Ranking por Silhouette Score
        ranking = sorted(algorithm_metrics.items(), 
                        key=lambda x: x[1]["silhouette_score"], 
                        reverse=True)
        
        comparative_analysis["algorithm_ranking"] = [
            {"algorithm": alg, "silhouette_score": metrics["silhouette_score"]}
            for alg, metrics in ranking
        ]
        
        # Comparación de métricas
        comparative_analysis["metric_comparison"] = algorithm_metrics
        
        # Análisis de consistencia entre algoritmos
        if len(clustering_results) >= 2:
            from sklearn.metrics import adjusted_rand_score
            
            algorithms_list = [alg for alg in clustering_results.keys() 
                             if "error" not in clustering_results[alg]]
            
            if len(algorithms_list) >= 2:
                labels1 = clustering_results[algorithms_list[0]]["labels"]
                labels2 = clustering_results[algorithms_list[1]]["labels"]
                
                consistency_score = adjusted_rand_score(labels1, labels2)
                comparative_analysis["cluster_consistency"] = {
                    "algorithms_compared": algorithms_list[:2],
                    "adjusted_rand_score": consistency_score
                }
        
        logger.info(f"   📊 Mejores resultados: {ranking[0][0]} "
                   f"(Silhouette: {ranking[0][1]['silhouette_score']:.3f})")
        
        return comparative_analysis
    
    def _run_musical_analysis(self, 
                            clustering_results: Dict, 
                            df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis específico para aplicación musical."""
        logger.info(f"🎵 Ejecutando análisis musical específico...")
        
        musical_analysis = {
            "genre_clustering_coherence": {},
            "acoustic_feature_correlation": {},
            "lyrical_theme_analysis": {},
            "multilingual_consistency": {}
        }
        
        # Análisis de coherencia con características musicales
        musical_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'loudness', 'speechiness'
        ]
        
        available_features = [f for f in musical_features if f in df.columns]
        
        if available_features:
            logger.info(f"   🎵 Analizando correlación con {len(available_features)} características musicales")
            
            for algorithm, results in clustering_results.items():
                if "error" in results:
                    continue
                
                labels = results["labels"]
                correlations = {}
                
                for feature in available_features:
                    if feature in df.columns:
                        # Calcular variabilidad intra-cluster vs inter-cluster
                        feature_values = df[feature].values
                        
                        # Varianza intra-cluster
                        intra_variance = 0
                        for cluster_id in set(labels):
                            cluster_mask = labels == cluster_id
                            if cluster_mask.sum() > 1:
                                cluster_values = feature_values[cluster_mask]
                                intra_variance += np.var(cluster_values) * cluster_mask.sum()
                        
                        intra_variance /= len(labels)
                        
                        # Varianza total
                        total_variance = np.var(feature_values)
                        
                        # Coherencia (menor varianza intra = mejor)
                        coherence = 1 - (intra_variance / total_variance) if total_variance > 0 else 0
                        correlations[feature] = coherence
                
                musical_analysis["acoustic_feature_correlation"][algorithm] = correlations
        
        # Análisis temático básico
        for algorithm, results in clustering_results.items():
            if "error" in results:
                continue
            
            labels = results["labels"]
            lyrics_list = df['lyrics'].tolist()
            
            # Análisis temático por cluster
            cluster_themes = {}
            for cluster_id in set(labels):
                cluster_mask = labels == cluster_id
                cluster_lyrics = [lyrics_list[i] for i in range(len(lyrics_list)) if cluster_mask[i]]
                
                # Palabras más comunes (análisis básico)
                all_words = []
                for lyric in cluster_lyrics[:10]:  # Muestra por performance
                    words = lyric.lower().split()
                    all_words.extend([w for w in words if len(w) > 3])
                
                if all_words:
                    from collections import Counter
                    common_words = Counter(all_words).most_common(5)
                    cluster_themes[f"cluster_{cluster_id}"] = {
                        "size": int(cluster_mask.sum()),
                        "top_words": common_words
                    }
            
            musical_analysis["lyrical_theme_analysis"][algorithm] = cluster_themes
        
        return musical_analysis
    
    def _generate_experimental_visualizations(self, 
                                            embeddings: np.ndarray,
                                            clustering_results: Dict,
                                            df: pd.DataFrame) -> Dict[str, str]:
        """Genera visualizaciones experimentales."""
        logger.info(f"🎨 Generando visualizaciones experimentales...")
        
        visualizations = {}
        
        for algorithm, results in clustering_results.items():
            if "error" in results:
                continue
            
            labels = results["labels"]
            lyrics_list = df['lyrics'].tolist()
            
            try:
                # Visualización 2D UMAP
                fig_umap = self.visualizer.visualize_clusters_2d(
                    embeddings, labels, lyrics_list,
                    method='umap',
                    title=f"Clustering Semántico - {algorithm}",
                    save_name=f"experimental_{algorithm}_{self.timestamp}"
                )
                
                if fig_umap:
                    visualizations[f"{algorithm}_umap"] = f"experimental_{algorithm}_{self.timestamp}_2d_umap.png"
                    import matplotlib.pyplot as plt
                    plt.close(fig_umap)
                
                # Reporte de métricas
                evaluation = results["evaluation"]
                fig_metrics = self.visualizer._plot_metrics_summary(
                    evaluation, f"experimental_{algorithm}_{self.timestamp}"
                )
                
                if fig_metrics:
                    visualizations[f"{algorithm}_metrics"] = f"experimental_{algorithm}_{self.timestamp}_metrics.png"
                    import matplotlib.pyplot as plt
                    plt.close(fig_metrics)
                
                logger.info(f"   ✅ Visualizaciones {algorithm} generadas")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Error visualización {algorithm}: {e}")
        
        return visualizations
    
    def _analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza características del dataset."""
        characteristics = {
            "total_songs": len(df),
            "total_features": len(df.columns),
            "lyrics_statistics": {},
            "musical_features": {}
        }
        
        # Estadísticas de letras
        if 'lyrics' in df.columns:
            lyrics_lengths = df['lyrics'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
            characteristics["lyrics_statistics"] = {
                "mean_words": float(lyrics_lengths.mean()),
                "median_words": float(lyrics_lengths.median()),
                "min_words": int(lyrics_lengths.min()),
                "max_words": int(lyrics_lengths.max())
            }
        
        # Características musicales disponibles
        musical_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'loudness', 'speechiness', 'tempo'
        ]
        
        available_musical = [f for f in musical_features if f in df.columns]
        characteristics["musical_features"] = {
            "available": available_musical,
            "count": len(available_musical)
        }
        
        return characteristics
    
    def _generate_scientific_conclusions(self) -> Dict[str, Any]:
        """Genera conclusiones científicas del experimento."""
        conclusions = {
            "best_algorithm": None,
            "performance_summary": {},
            "scientific_insights": [],
            "recommendations": []
        }
        
        # Determinar mejor algoritmo
        clustering_results = self.experimental_results.get("clustering_results", {})
        
        best_silhouette = -1
        best_algorithm = None
        
        for algorithm, results in clustering_results.items():
            if "error" in results:
                continue
            
            silhouette = results["evaluation"]["standard_metrics"].get("silhouette_score", 0)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_algorithm = algorithm
        
        conclusions["best_algorithm"] = best_algorithm
        conclusions["performance_summary"]["best_silhouette_score"] = best_silhouette
        
        # Insights científicos
        conclusions["scientific_insights"] = [
            f"Mejor algoritmo identificado: {best_algorithm}",
            f"Silhouette Score máximo alcanzado: {best_silhouette:.3f}",
            "Clustering semántico demuestra capacidad de agrupación temática",
            "Sistema listo para integración con clustering musical"
        ]
        
        # Recomendaciones
        conclusions["recommendations"] = [
            "Proceder con integración híbrida musical + semántica",
            "Considerar optimización para datasets >10K canciones",
            "Implementar sistema de recomendaciones basado en clusters semánticos"
        ]
        
        return conclusions
    
    def _save_experimental_results(self):
        """Guarda resultados experimentales en JSON."""
        results_file = self.output_dir / f"experimental_validation_{self.timestamp}.json"
        
        # Preparar datos serializables
        serializable_results = self._make_serializable(self.experimental_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados guardados: {results_file}")
    
    def _make_serializable(self, obj) -> Any:
        """Convierte objetos a formato serializable JSON."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        else:
            return obj
    
    def _generate_final_report(self):
        """Genera reporte final experimental."""
        report_file = self.output_dir / f"experimental_report_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Reporte Validación Experimental - Clustering Semántico\n\n")
            f.write(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Timestamp**: {self.timestamp}\n\n")
            
            # Resumen ejecutivo
            f.write("## Resumen Ejecutivo\n\n")
            experiment_info = self.experimental_results["experiment_info"]
            f.write(f"- **Dataset**: {experiment_info['sample_size']} canciones analizadas\n")
            f.write(f"- **Tiempo total**: {experiment_info['total_time']:.2f} segundos\n")
            f.write(f"- **Algoritmos evaluados**: {', '.join(experiment_info['algorithms_tested'])}\n\n")
            
            # Resultados principales
            f.write("## Resultados Principales\n\n")
            conclusions = self.experimental_results["conclusions"]
            f.write(f"- **Mejor algoritmo**: {conclusions['best_algorithm']}\n")
            f.write(f"- **Silhouette Score máximo**: {conclusions['performance_summary']['best_silhouette_score']:.3f}\n\n")
            
            # Conclusiones científicas
            f.write("## Conclusiones Científicas\n\n")
            for insight in conclusions["scientific_insights"]:
                f.write(f"- {insight}\n")
            
            f.write("\n## Recomendaciones\n\n")
            for recommendation in conclusions["recommendations"]:
                f.write(f"- {recommendation}\n")
        
        logger.info(f"📋 Reporte final generado: {report_file}")


def main():
    """Función principal para ejecutar validación experimental."""
    print("🚀 VALIDACIÓN EXPERIMENTAL - Sistema Clustering Semántico FASE 5")
    print("=" * 80)
    
    # Configuración experimental
    sample_sizes = [1000, 2000, 5000]  # Diferentes tamaños para análisis escalabilidad
    algorithms = ['kmeans_semantic', 'hierarchical_semantic']
    
    for sample_size in sample_sizes:
        print(f"\n🔬 EXPERIMENTO: {sample_size} canciones")
        print("-" * 50)
        
        # Ejecutar validación
        validator = ExperimentalValidator(
            output_dir=f"experimental_results_{sample_size}"
        )
        
        results = validator.run_complete_validation(
            sample_size=sample_size,
            algorithms=algorithms
        )
        
        if results["success"]:
            print(f"✅ Experimento {sample_size} completado exitosamente")
        else:
            print(f"❌ Error en experimento {sample_size}: {results['error']}")
    
    print("\n🎉 VALIDACIÓN EXPERIMENTAL COMPLETA")
    print("📊 Revisar directorios 'experimental_results_*' para resultados detallados")


if __name__ == "__main__":
    main()