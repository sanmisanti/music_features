#!/usr/bin/env python3
"""
Sistema de validación de interpretabilidad automática para FASE 3.
Genera etiquetas automáticas y valida coherencia temática de clusters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import logging

@dataclass
class ClusterInterpretation:
    """Interpretación automática de un cluster."""
    cluster_id: int
    cluster_size: int
    domain: str
    
    # Etiqueta automática
    automatic_label: str
    confidence_score: float
    
    # Características dominantes (musical)
    dominant_features: Optional[Dict[str, float]] = None
    feature_consistency: Optional[Dict[str, float]] = None
    
    # Coherencia semántica (semantic)
    avg_cosine_similarity: Optional[float] = None
    coherence_score: Optional[float] = None
    
    # Estadísticas del cluster
    cluster_stats: Optional[Dict[str, Any]] = None

class InterpretabilityValidator:
    """Validador de interpretabilidad automática para clusters."""
    
    def __init__(self, verbose: bool = True):
        """Inicializar validador."""
        self.verbose = verbose
        self.logger = self._setup_logging()
        
        # Nombres de características musicales (12D)
        self.musical_feature_names = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        # Mapeo de características a interpretaciones humanas
        self.feature_interpretations = {
            'danceability': {'high': 'Muy Bailable', 'low': 'Poco Bailable'},
            'energy': {'high': 'Alta Energía', 'low': 'Baja Energía'},
            'valence': {'high': 'Positivo/Alegre', 'low': 'Melancólico/Triste'},
            'acousticness': {'high': 'Acústico', 'low': 'Electrónico'},
            'instrumentalness': {'high': 'Instrumental', 'low': 'Vocal'},
            'liveness': {'high': 'En Vivo', 'low': 'Estudio'},
            'speechiness': {'high': 'Hablado', 'low': 'Musical'},
            'loudness': {'high': 'Fuerte', 'low': 'Suave'},
            'tempo': {'high': 'Rápido', 'low': 'Lento'},
            'duration_ms': {'high': 'Canción Larga', 'low': 'Canción Corta'}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar logging."""
        logger = logging.getLogger('InterpretabilityValidator')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[INTERPRETABILITY] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_musical_cluster_interpretability(self, X: np.ndarray, 
                                                labels: np.ndarray) -> List[ClusterInterpretation]:
        """
        Validar interpretabilidad de clusters musicales.
        
        Args:
            X: Características musicales (N, 12)
            labels: Etiquetas de clustering
            
        Returns:
            Lista de interpretaciones de clusters
        """
        self.logger.info("Validando interpretabilidad de clusters musicales")
        
        unique_labels = np.unique(labels[labels != -1])  # Filtrar outliers
        interpretations = []
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = len(cluster_data)
            
            if cluster_size < 2:
                continue
            
            # Calcular estadísticas del cluster
            cluster_means = np.mean(cluster_data, axis=0)
            cluster_stds = np.std(cluster_data, axis=0)
            
            # Identificar características dominantes
            dominant_features = {}
            feature_consistency = {}
            
            for i, feature_name in enumerate(self.musical_feature_names):
                mean_val = cluster_means[i]
                std_val = cluster_stds[i]
                
                dominant_features[feature_name] = float(mean_val)
                
                # Consistencia = 1 / (1 + coeficiente de variación)
                cv = std_val / (abs(mean_val) + 1e-10)
                consistency = 1.0 / (1.0 + cv)
                feature_consistency[feature_name] = float(consistency)
            
            # Generar etiqueta automática basada en características más distintivas
            automatic_label, confidence = self._generate_musical_label(
                dominant_features, feature_consistency
            )
            
            # Calcular estadísticas generales
            cluster_stats = {
                'mean_values': {k: float(v) for k, v in zip(self.musical_feature_names, cluster_means)},
                'std_values': {k: float(v) for k, v in zip(self.musical_feature_names, cluster_stds)},
                'consistency_scores': feature_consistency
            }
            
            interpretation = ClusterInterpretation(
                cluster_id=int(cluster_id),
                cluster_size=cluster_size,
                domain='musical',
                automatic_label=automatic_label,
                confidence_score=confidence,
                dominant_features=dominant_features,
                feature_consistency=feature_consistency,
                cluster_stats=cluster_stats
            )
            
            interpretations.append(interpretation)
        
        self.logger.info(f"Generadas {len(interpretations)} interpretaciones musicales")
        return interpretations
    
    def _generate_musical_label(self, dominant_features: Dict[str, float],
                              consistency: Dict[str, float]) -> Tuple[str, float]:
        """Generar etiqueta automática para cluster musical."""
        
        # Identificar las 3 características más consistentes y distintivas
        # Peso = consistencia × |valor normalizado - 0.5| (distancia del centro)
        feature_scores = {}
        
        for feature_name, mean_val in dominant_features.items():
            if feature_name in consistency:
                # Normalizar valor a [0,1] aproximadamente
                if feature_name == 'key':
                    norm_val = mean_val / 11.0  # Keys 0-11
                elif feature_name == 'mode':
                    norm_val = mean_val  # Ya 0-1
                elif feature_name == 'loudness':
                    norm_val = (mean_val + 60) / 60  # Aproximadamente [-60, 0]
                elif feature_name == 'tempo':
                    norm_val = min(mean_val / 200.0, 1.0)  # Hasta ~200 BPM
                elif feature_name == 'duration_ms':
                    norm_val = min(mean_val / 300000.0, 1.0)  # Hasta 5 minutos
                else:
                    norm_val = np.clip(mean_val, 0, 1)  # Ya debería estar [0,1]
                
                # Score = consistencia × distintividad
                distinctiveness = abs(norm_val - 0.5)  # Qué tan lejos del centro
                score = consistency[feature_name] * distinctiveness
                feature_scores[feature_name] = (score, norm_val, consistency[feature_name])
        
        # Ordenar por score y tomar top 3
        top_features = sorted(feature_scores.items(), 
                            key=lambda x: x[1][0], reverse=True)[:3]
        
        # Generar etiqueta descriptiva
        label_parts = []
        total_confidence = 0
        
        for feature_name, (score, norm_val, consist) in top_features:
            if feature_name in self.feature_interpretations:
                # Determinar si es "high" o "low"
                threshold = 0.6 if feature_name in ['energy', 'valence', 'danceability'] else 0.5
                
                if norm_val > threshold:
                    interpretation = self.feature_interpretations[feature_name]['high']
                else:
                    interpretation = self.feature_interpretations[feature_name]['low']
                
                label_parts.append(interpretation)
                total_confidence += consist
        
        # Combinar partes de la etiqueta
        if len(label_parts) >= 2:
            automatic_label = f"{label_parts[0]} & {label_parts[1]}"
        elif len(label_parts) == 1:
            automatic_label = label_parts[0]
        else:
            automatic_label = "Cluster Genérico"
        
        confidence_score = total_confidence / len(top_features) if top_features else 0.0
        
        return automatic_label, confidence_score
    
    def validate_semantic_cluster_interpretability(self, X: np.ndarray,
                                                 labels: np.ndarray) -> List[ClusterInterpretation]:
        """
        Validar interpretabilidad de clusters semánticos.
        
        Args:
            X: Embeddings semánticos (N, 384)
            labels: Etiquetas de clustering
            
        Returns:
            Lista de interpretaciones de clusters
        """
        self.logger.info("Validando interpretabilidad de clusters semánticos")
        
        unique_labels = np.unique(labels[labels != -1])  # Filtrar outliers
        interpretations = []
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = len(cluster_data)
            
            if cluster_size < 2:
                continue
            
            # Calcular similitud coseno interna
            similarity_matrix = cosine_similarity(cluster_data)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            avg_cosine_similarity = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0
            coherence_score = avg_cosine_similarity  # Para clusters semánticos, coherencia = similitud coseno
            
            # Generar etiqueta automática basada en coherencia
            automatic_label, confidence = self._generate_semantic_label(
                cluster_id, cluster_size, coherence_score
            )
            
            # Estadísticas del cluster
            cluster_stats = {
                'avg_cosine_similarity': avg_cosine_similarity,
                'min_similarity': float(np.min(upper_triangle)) if len(upper_triangle) > 0 else 0.0,
                'max_similarity': float(np.max(upper_triangle)) if len(upper_triangle) > 0 else 0.0,
                'std_similarity': float(np.std(upper_triangle)) if len(upper_triangle) > 0 else 0.0,
                'embedding_centroid_norm': float(np.linalg.norm(np.mean(cluster_data, axis=0)))
            }
            
            interpretation = ClusterInterpretation(
                cluster_id=int(cluster_id),
                cluster_size=cluster_size,
                domain='semantic',
                automatic_label=automatic_label,
                confidence_score=confidence,
                avg_cosine_similarity=avg_cosine_similarity,
                coherence_score=coherence_score,
                cluster_stats=cluster_stats
            )
            
            interpretations.append(interpretation)
        
        self.logger.info(f"Generadas {len(interpretations)} interpretaciones semánticas")
        return interpretations
    
    def _generate_semantic_label(self, cluster_id: int, cluster_size: int,
                               coherence_score: float) -> Tuple[str, float]:
        """Generar etiqueta automática para cluster semántico."""
        
        # Etiquetas basadas en coherencia y tamaño
        if coherence_score > 0.8:
            coherence_desc = "Muy Coherente"
        elif coherence_score > 0.6:
            coherence_desc = "Coherente"
        elif coherence_score > 0.4:
            coherence_desc = "Moderadamente Coherente"
        else:
            coherence_desc = "Poco Coherente"
        
        # Descripción de tamaño
        if cluster_size > 1000:
            size_desc = "Tema Principal"
        elif cluster_size > 500:
            size_desc = "Tema Importante"
        elif cluster_size > 100:
            size_desc = "Subtema"
        else:
            size_desc = "Tema Específico"
        
        # Combinar descripciones
        automatic_label = f"{size_desc} {coherence_desc}"
        
        # Confidence basada en coherencia y tamaño relativo
        size_confidence = min(cluster_size / 1000.0, 1.0)  # Normalizar tamaño
        confidence_score = (coherence_score + size_confidence) / 2.0
        
        return automatic_label, confidence_score
    
    def validate_all_clusters(self, X_musical: np.ndarray, labels_musical: np.ndarray,
                            X_semantic: np.ndarray, labels_semantic: np.ndarray) -> Dict[str, List[ClusterInterpretation]]:
        """
        Validar interpretabilidad para ambos dominios.
        
        Args:
            X_musical: Características musicales
            labels_musical: Etiquetas clustering musical
            X_semantic: Embeddings semánticos
            labels_semantic: Etiquetas clustering semántico
            
        Returns:
            Dict con interpretaciones por dominio
        """
        self.logger.info("Validando interpretabilidad de todos los clusters")
        
        # Validar clusters musicales
        musical_interpretations = self.validate_musical_cluster_interpretability(
            X_musical, labels_musical
        )
        
        # Validar clusters semánticos
        semantic_interpretations = self.validate_semantic_cluster_interpretability(
            X_semantic, labels_semantic
        )
        
        results = {
            'musical': musical_interpretations,
            'semantic': semantic_interpretations
        }
        
        # Estadísticas generales
        total_musical = len([i for i in musical_interpretations if i.confidence_score > 0.5])
        total_semantic = len([i for i in semantic_interpretations if i.confidence_score > 0.5])
        
        self.logger.info(f"Clusters interpretables: Musical {total_musical}/{len(musical_interpretations)}, "
                        f"Semántico {total_semantic}/{len(semantic_interpretations)}")
        
        return results
    
    def export_interpretations_to_dataframe(self, interpretations: List[ClusterInterpretation]) -> pd.DataFrame:
        """Exportar interpretaciones a DataFrame."""
        rows = []
        
        for interp in interpretations:
            row = {
                'cluster_id': interp.cluster_id,
                'cluster_size': interp.cluster_size,
                'domain': interp.domain,
                'automatic_label': interp.automatic_label,
                'confidence_score': interp.confidence_score
            }
            
            if interp.domain == 'musical' and interp.dominant_features:
                row.update(interp.dominant_features)
                if interp.feature_consistency:
                    for k, v in interp.feature_consistency.items():
                        row[f'{k}_consistency'] = v
                        
            elif interp.domain == 'semantic':
                row['avg_cosine_similarity'] = interp.avg_cosine_similarity
                row['coherence_score'] = interp.coherence_score
            
            rows.append(row)
        
        return pd.DataFrame(rows)