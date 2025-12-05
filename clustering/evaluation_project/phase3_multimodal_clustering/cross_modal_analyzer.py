#!/usr/bin/env python3
"""
Analizador de correspondencias cross-modales para FASE 3.
Evalúa coherencia entre clustering musical y semántico de las mismas canciones.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class CrossModalCorrespondence:
    """Correspondencia entre clusters de diferentes dominios."""
    musical_cluster_id: int
    semantic_cluster_id: int
    overlap_count: int
    overlap_ratio: float
    musical_cluster_size: int
    semantic_cluster_size: int
    correspondence_strength: float

@dataclass
class CrossModalAnalysisResult:
    """Resultado completo de análisis cross-modal."""
    nmi_score: float
    adjusted_rand_score: float
    n_musical_clusters: int
    n_semantic_clusters: int
    n_valid_samples: int
    
    # Correspondencias principales
    strong_correspondences: List[CrossModalCorrespondence]
    weak_correspondences: List[CrossModalCorrespondence]
    
    # Matriz de contingencia
    contingency_matrix: np.ndarray
    musical_cluster_sizes: Dict[int, int]
    semantic_cluster_sizes: Dict[int, int]
    
    # Métricas de coherencia
    avg_correspondence_strength: float
    correspondence_coverage: float  # % de muestras en correspondencias fuertes
    
    # Análisis de divergencias
    divergent_cases: Dict[str, Any]

class CrossModalAnalyzer:
    """Analizador de correspondencias cross-modales."""
    
    def __init__(self, verbose: bool = True):
        """Inicializar analizador."""
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Umbrales para correspondencias
        self.strong_correspondence_threshold = 0.3  # 30% de overlap
        self.weak_correspondence_threshold = 0.1    # 10% de overlap
        
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging."""
        logger = logging.getLogger('CrossModalAnalyzer')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[CROSS-MODAL] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_cross_modal_correspondence(self, labels_musical: np.ndarray,
                                         labels_semantic: np.ndarray,
                                         track_ids: Optional[np.ndarray] = None) -> CrossModalAnalysisResult:
        """
        Analizar correspondencias cross-modales entre clusterings.
        
        Args:
            labels_musical: Etiquetas clustering musical
            labels_semantic: Etiquetas clustering semántico
            track_ids: IDs de canciones (opcional, para debugging)
            
        Returns:
            Resultado completo del análisis cross-modal
        """
        self.logger.info("Iniciando análisis cross-modal")
        
        # Validar entrada
        if len(labels_musical) != len(labels_semantic):
            raise ValueError("Las etiquetas deben tener la misma longitud")
        
        # Filtrar outliers de ambos dominios
        valid_mask = (labels_musical != -1) & (labels_semantic != -1)
        n_valid_samples = np.sum(valid_mask)
        
        if n_valid_samples < 10:
            self.logger.warning(f"Muy pocas muestras válidas: {n_valid_samples}")
            return self._create_empty_result()
        
        labels_musical_valid = labels_musical[valid_mask]
        labels_semantic_valid = labels_semantic[valid_mask]
        
        self.logger.info(f"Muestras válidas para análisis: {n_valid_samples}/{len(labels_musical)}")
        
        # Calcular métricas de correspondencia global
        nmi_score = normalized_mutual_info_score(labels_musical_valid, labels_semantic_valid)
        ari_score = adjusted_rand_score(labels_musical_valid, labels_semantic_valid)
        
        self.logger.info(f"NMI Score: {nmi_score:.3f}, ARI Score: {ari_score:.3f}")
        
        # Obtener información de clusters
        unique_musical = np.unique(labels_musical_valid)
        unique_semantic = np.unique(labels_semantic_valid)
        
        n_musical_clusters = len(unique_musical)
        n_semantic_clusters = len(unique_semantic)
        
        # Calcular tamaños de clusters
        musical_cluster_sizes = dict(Counter(labels_musical_valid))
        semantic_cluster_sizes = dict(Counter(labels_semantic_valid))
        
        # Crear matriz de contingencia
        contingency_matrix = confusion_matrix(labels_musical_valid, labels_semantic_valid,
                                            labels=unique_musical)
        
        # Analizar correspondencias específicas
        correspondences = self._analyze_correspondences(
            contingency_matrix, unique_musical, unique_semantic,
            musical_cluster_sizes, semantic_cluster_sizes, n_valid_samples
        )
        
        # Clasificar correspondencias por fuerza
        strong_correspondences = [c for c in correspondences 
                                if c.correspondence_strength >= self.strong_correspondence_threshold]
        weak_correspondences = [c for c in correspondences 
                              if self.weak_correspondence_threshold <= c.correspondence_strength < self.strong_correspondence_threshold]
        
        # Calcular métricas de coherencia
        avg_correspondence_strength = np.mean([c.correspondence_strength for c in correspondences]) if correspondences else 0.0
        
        # Calcular cobertura de correspondencias fuertes
        strong_coverage_count = sum(c.overlap_count for c in strong_correspondences)
        correspondence_coverage = strong_coverage_count / n_valid_samples if n_valid_samples > 0 else 0.0
        
        # Analizar casos divergentes
        divergent_cases = self._analyze_divergent_cases(
            contingency_matrix, unique_musical, unique_semantic,
            musical_cluster_sizes, semantic_cluster_sizes
        )
        
        result = CrossModalAnalysisResult(
            nmi_score=float(nmi_score),
            adjusted_rand_score=float(ari_score),
            n_musical_clusters=n_musical_clusters,
            n_semantic_clusters=n_semantic_clusters,
            n_valid_samples=n_valid_samples,
            strong_correspondences=strong_correspondences,
            weak_correspondences=weak_correspondences,
            contingency_matrix=contingency_matrix,
            musical_cluster_sizes=musical_cluster_sizes,
            semantic_cluster_sizes=semantic_cluster_sizes,
            avg_correspondence_strength=float(avg_correspondence_strength),
            correspondence_coverage=float(correspondence_coverage),
            divergent_cases=divergent_cases
        )
        
        self._log_analysis_summary(result)
        
        return result
    
    def _analyze_correspondences(self, contingency_matrix: np.ndarray,
                               unique_musical: np.ndarray, unique_semantic: np.ndarray,
                               musical_sizes: Dict[int, int], semantic_sizes: Dict[int, int],
                               total_samples: int) -> List[CrossModalCorrespondence]:
        """Analizar correspondencias específicas entre clusters."""
        correspondences = []
        
        for i, musical_cluster in enumerate(unique_musical):
            for j, semantic_cluster in enumerate(unique_semantic):
                overlap_count = int(contingency_matrix[i, j])
                
                if overlap_count == 0:
                    continue
                
                musical_cluster_size = musical_sizes[musical_cluster]
                semantic_cluster_size = semantic_sizes[semantic_cluster]
                
                # Ratio de overlap respecto al cluster más pequeño
                min_cluster_size = min(musical_cluster_size, semantic_cluster_size)
                overlap_ratio = overlap_count / min_cluster_size
                
                # Fuerza de correspondencia basada en múltiples factores
                # Factor 1: Ratio de overlap
                overlap_strength = overlap_ratio
                
                # Factor 2: Significancia estadística (overlap vs esperado por azar)
                expected_overlap = (musical_cluster_size * semantic_cluster_size) / total_samples
                statistical_significance = overlap_count / (expected_overlap + 1e-10)
                
                # Combinear factores
                correspondence_strength = (overlap_strength + min(statistical_significance / 5.0, 1.0)) / 2.0
                
                correspondence = CrossModalCorrespondence(
                    musical_cluster_id=int(musical_cluster),
                    semantic_cluster_id=int(semantic_cluster),
                    overlap_count=overlap_count,
                    overlap_ratio=float(overlap_ratio),
                    musical_cluster_size=musical_cluster_size,
                    semantic_cluster_size=semantic_cluster_size,
                    correspondence_strength=float(correspondence_strength)
                )
                
                correspondences.append(correspondence)
        
        # Ordenar por fuerza de correspondencia
        correspondences.sort(key=lambda c: c.correspondence_strength, reverse=True)
        
        return correspondences
    
    def _analyze_divergent_cases(self, contingency_matrix: np.ndarray,
                               unique_musical: np.ndarray, unique_semantic: np.ndarray,
                               musical_sizes: Dict[int, int], semantic_sizes: Dict[int, int]) -> Dict[str, Any]:
        """Analizar casos donde los dominios divergen significativamente."""
        
        # Clusters musicales con alta fragmentación semántica
        musical_fragmented = []
        for i, musical_cluster in enumerate(unique_musical):
            row = contingency_matrix[i, :]
            n_semantic_clusters_involved = np.sum(row > 0)
            musical_size = musical_sizes[musical_cluster]
            
            if n_semantic_clusters_involved > 3 and musical_size > 50:
                fragmentation_ratio = n_semantic_clusters_involved / musical_size
                musical_fragmented.append({
                    'musical_cluster_id': int(musical_cluster),
                    'cluster_size': musical_size,
                    'semantic_clusters_involved': int(n_semantic_clusters_involved),
                    'fragmentation_ratio': float(fragmentation_ratio)
                })
        
        # Clusters semánticos con alta fragmentación musical
        semantic_fragmented = []
        for j, semantic_cluster in enumerate(unique_semantic):
            col = contingency_matrix[:, j]
            n_musical_clusters_involved = np.sum(col > 0)
            semantic_size = semantic_sizes[semantic_cluster]
            
            if n_musical_clusters_involved > 3 and semantic_size > 50:
                fragmentation_ratio = n_musical_clusters_involved / semantic_size
                semantic_fragmented.append({
                    'semantic_cluster_id': int(semantic_cluster),
                    'cluster_size': semantic_size,
                    'musical_clusters_involved': int(n_musical_clusters_involved),
                    'fragmentation_ratio': float(fragmentation_ratio)
                })
        
        return {
            'musical_fragmented': musical_fragmented,
            'semantic_fragmented': semantic_fragmented,
            'n_musical_fragmented': len(musical_fragmented),
            'n_semantic_fragmented': len(semantic_fragmented)
        }
    
    def _create_empty_result(self) -> CrossModalAnalysisResult:
        """Crear resultado vacío para casos con datos insuficientes."""
        return CrossModalAnalysisResult(
            nmi_score=0.0,
            adjusted_rand_score=0.0,
            n_musical_clusters=0,
            n_semantic_clusters=0,
            n_valid_samples=0,
            strong_correspondences=[],
            weak_correspondences=[],
            contingency_matrix=np.array([[]]),
            musical_cluster_sizes={},
            semantic_cluster_sizes={},
            avg_correspondence_strength=0.0,
            correspondence_coverage=0.0,
            divergent_cases={'musical_fragmented': [], 'semantic_fragmented': []}
        )
    
    def _log_analysis_summary(self, result: CrossModalAnalysisResult):
        """Log resumen del análisis."""
        self.logger.info(f"=== RESUMEN ANÁLISIS CROSS-MODAL ===")
        self.logger.info(f"NMI Score: {result.nmi_score:.3f}")
        self.logger.info(f"Clusters Musical: {result.n_musical_clusters}, Semántico: {result.n_semantic_clusters}")
        self.logger.info(f"Correspondencias fuertes: {len(result.strong_correspondences)}")
        self.logger.info(f"Correspondencias débiles: {len(result.weak_correspondences)}")
        self.logger.info(f"Cobertura correspondencias fuertes: {result.correspondence_coverage:.1%}")
        self.logger.info(f"Fuerza promedio: {result.avg_correspondence_strength:.3f}")
    
    def visualize_contingency_matrix(self, result: CrossModalAnalysisResult, 
                                   output_path: Optional[Path] = None,
                                   title: str = "Cross-Modal Correspondence Matrix") -> str:
        """
        Visualizar matriz de contingencia cross-modal.
        
        Args:
            result: Resultado de análisis cross-modal
            output_path: Ruta de salida (opcional)
            title: Título del gráfico
            
        Returns:
            Path del archivo guardado
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Crear heatmap
        sns.heatmap(result.contingency_matrix, 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f'S{i}' for i in range(result.n_semantic_clusters)],
                   yticklabels=[f'M{i}' for i in range(result.n_musical_clusters)],
                   ax=ax)
        
        ax.set_title(f'{title}\nNMI: {result.nmi_score:.3f}, Strong Correspondences: {len(result.strong_correspondences)}')
        ax.set_xlabel('Semantic Clusters')
        ax.set_ylabel('Musical Clusters')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Matriz de contingencia guardada: {output_path}")
            plt.close()
            return str(output_path)
        else:
            plt.show()
            return ""
    
    def export_correspondences_to_dataframe(self, correspondences: List[CrossModalCorrespondence]) -> pd.DataFrame:
        """Exportar correspondencias a DataFrame."""
        if not correspondences:
            return pd.DataFrame()
        
        rows = []
        for corr in correspondences:
            rows.append({
                'musical_cluster_id': corr.musical_cluster_id,
                'semantic_cluster_id': corr.semantic_cluster_id,
                'overlap_count': corr.overlap_count,
                'overlap_ratio': corr.overlap_ratio,
                'musical_cluster_size': corr.musical_cluster_size,
                'semantic_cluster_size': corr.semantic_cluster_size,
                'correspondence_strength': corr.correspondence_strength
            })
        
        return pd.DataFrame(rows)
    
    def generate_cross_modal_report(self, result: CrossModalAnalysisResult) -> str:
        """Generar reporte textual del análisis cross-modal."""
        
        report_lines = [
            "# REPORTE ANÁLISIS CROSS-MODAL",
            "",
            "## Métricas Globales",
            f"- **NMI Score**: {result.nmi_score:.3f}",
            f"- **Adjusted Rand Index**: {result.adjusted_rand_score:.3f}",
            f"- **Muestras válidas**: {result.n_valid_samples:,}",
            f"- **Clusters musicales**: {result.n_musical_clusters}",
            f"- **Clusters semánticos**: {result.n_semantic_clusters}",
            "",
            "## Correspondencias",
            f"- **Correspondencias fuertes**: {len(result.strong_correspondences)} (threshold ≥ {self.strong_correspondence_threshold})",
            f"- **Correspondencias débiles**: {len(result.weak_correspondences)} (threshold ≥ {self.weak_correspondence_threshold})",
            f"- **Cobertura correspondencias fuertes**: {result.correspondence_coverage:.1%}",
            f"- **Fuerza promedio**: {result.avg_correspondence_strength:.3f}",
            ""
        ]
        
        # Detallar correspondencias fuertes
        if result.strong_correspondences:
            report_lines.extend([
                "## Correspondencias Fuertes",
                ""
            ])
            
            for i, corr in enumerate(result.strong_correspondences[:5]):  # Top 5
                report_lines.append(
                    f"{i+1}. Musical Cluster {corr.musical_cluster_id} ↔ Semantic Cluster {corr.semantic_cluster_id}: "
                    f"{corr.overlap_count} canciones ({corr.overlap_ratio:.1%} overlap, "
                    f"strength={corr.correspondence_strength:.3f})"
                )
        
        # Casos divergentes
        if result.divergent_cases['musical_fragmented'] or result.divergent_cases['semantic_fragmented']:
            report_lines.extend([
                "",
                "## Casos Divergentes",
                f"- **Clusters musicales fragmentados**: {result.divergent_cases['n_musical_fragmented']}",
                f"- **Clusters semánticos fragmentados**: {result.divergent_cases['n_semantic_fragmented']}",
            ])
        
        return "\n".join(report_lines)