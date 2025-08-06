"""
Clustering Readiness Assessment Module

Este módulo evalúa qué tan adecuado es un dataset para clustering efectivo,
proporcionando métricas específicas y recomendaciones para optimización.

Funcionalidades:
- Hopkins Statistic para evaluar clustering tendency
- Recomendación de número óptimo de clusters (K)
- Análisis de separabilidad y calidad esperada
- Feature selection para clustering
- Diagnóstico de problemas y soluciones
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class ClusteringReadiness:
    """
    Evaluador de preparación para clustering de datasets musicales.
    
    Este módulo es crítico para determinar si un dataset es adecuado
    para clustering y qué estrategias usar para optimizar los resultados.
    """
    
    def __init__(self, musical_features=None):
        """
        Inicializar evaluador de clustering readiness.
        
        Args:
            musical_features: Lista de características musicales a analizar
        """
        self.musical_features = musical_features or [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
            'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
        
    def assess_clustering_tendency(self, df):
        """
        Evaluar si los datos tienen tendencia natural al clustering.
        
        Calcula Hopkins Statistic para determinar si los datos son
        clusterizables (>0.5) o aleatorios (<0.5).
        
        Args:
            df: DataFrame con características musicales
            
        Returns:
            dict: Resultados de clustering tendency assessment
        """
        try:
            # Preparar datos
            X = self._prepare_data(df)
            if X is None:
                return self._create_error_result("Error preparando datos")
            
            # Calcular Hopkins Statistic
            hopkins_stat = self._calculate_hopkins_statistic(X)
            
            # Interpretación
            if hopkins_stat > 0.75:
                interpretation = "EXCELENTE - Datos altamente clusterizables"
                confidence = "muy alta"
            elif hopkins_stat > 0.6:
                interpretation = "BUENO - Datos moderadamente clusterizables" 
                confidence = "alta"
            elif hopkins_stat > 0.5:
                interpretation = "ACEPTABLE - Clustering posible con optimización"
                confidence = "moderada"
            else:
                interpretation = "PROBLEMÁTICO - Datos tienden a ser aleatorios"
                confidence = "baja"
            
            return {
                'hopkins_statistic': hopkins_stat,
                'is_clusterable': hopkins_stat > 0.5,
                'confidence_score': hopkins_stat,
                'interpretation': interpretation,
                'confidence_level': confidence,
                'sample_size': len(X),
                'n_features': X.shape[1]
            }
            
        except Exception as e:
            return self._create_error_result(f"Error en clustering tendency: {str(e)}")
    
    def recommend_optimal_k(self, df, k_range=(2, 15)):
        """
        Determinar número óptimo de clusters usando múltiples métodos.
        
        Combina Elbow Method, Silhouette Score y Calinski-Harabasz
        para recomendar el K más apropiado.
        
        Args:
            df: DataFrame con características musicales
            k_range: Tuple con rango de K a evaluar
            
        Returns:
            dict: Recomendación de K óptimo y métricas
        """
        try:
            # Preparar datos
            X = self._prepare_data(df)
            if X is None:
                return self._create_error_result("Error preparando datos")
                
            # Limitar dataset si es muy grande (performance)
            if len(X) > 5000:
                indices = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            k_min, k_max = k_range
            k_values = range(k_min, min(k_max + 1, len(X_sample) // 2))
            
            metrics = {
                'k_values': list(k_values),
                'inertias': [],
                'silhouette_scores': [],
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': []
            }
            
            # Evaluar cada K
            for k in k_values:
                try:
                    # K-Means clustering
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_sample)
                    
                    # Métricas
                    metrics['inertias'].append(kmeans.inertia_)
                    
                    if len(np.unique(labels)) > 1:  # Al menos 2 clusters
                        silhouette = silhouette_score(X_sample, labels)
                        calinski = calinski_harabasz_score(X_sample, labels)
                        davies_bouldin = davies_bouldin_score(X_sample, labels)
                        
                        metrics['silhouette_scores'].append(silhouette)
                        metrics['calinski_harabasz_scores'].append(calinski)
                        metrics['davies_bouldin_scores'].append(davies_bouldin)
                    else:
                        # Cluster único - métricas no válidas
                        metrics['silhouette_scores'].append(-1.0)
                        metrics['calinski_harabasz_scores'].append(0.0)
                        metrics['davies_bouldin_scores'].append(10.0)
                        
                except Exception as e:
                    print(f"Error evaluando K={k}: {e}")
                    continue
            
            # Encontrar K óptimo
            optimal_k_results = self._find_optimal_k(metrics)
            
            return {
                'recommended_k': optimal_k_results['best_k'],
                'k_range_probable': optimal_k_results['k_range'],
                'methods_agreement': optimal_k_results['agreement'],
                'quality_preview': optimal_k_results['quality'],
                'all_metrics': metrics,
                'evaluation_notes': optimal_k_results['notes']
            }
            
        except Exception as e:
            return self._create_error_result(f"Error en K optimization: {str(e)}")
    
    def analyze_cluster_separability(self, df):
        """
        Evaluar qué tan separables serán los clusters.
        
        Analiza la distribución de distancias entre puntos para
        predecir la calidad de separación de clusters.
        
        Args:
            df: DataFrame con características musicales
            
        Returns:
            dict: Análisis de separabilidad esperada
        """
        try:
            # Preparar datos
            X = self._prepare_data(df)
            if X is None:
                return self._create_error_result("Error preparando datos")
            
            # Muestra para performance si dataset es grande
            if len(X) > 2000:
                indices = np.random.choice(len(X), 2000, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Calcular distancias entre todos los puntos
            distances = pdist(X_sample, metric='euclidean')
            
            # Estadísticas de distancias
            distance_stats = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'distance_range': np.max(distances) - np.min(distances)
            }
            
            # Análisis de densidad (k-nearest neighbors)
            k = min(10, len(X_sample) // 4)  # k adaptativo
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
            distances_knn, _ = nbrs.kneighbors(X_sample)
            avg_knn_distance = np.mean(distances_knn[:, 1:])  # Excluir distancia a sí mismo
            
            # Calcular separabilidad score
            separability_score = self._calculate_separability_score(distance_stats, avg_knn_distance)
            
            # Predecir rango de Silhouette esperado
            expected_silhouette = self._predict_silhouette_range(separability_score, distance_stats)
            
            # Detectar riesgo de overlap
            overlap_risk = self._assess_overlap_risk(distances, avg_knn_distance)
            
            return {
                'separability_score': separability_score,
                'expected_silhouette_range': expected_silhouette,
                'overlap_risk': overlap_risk,
                'distance_distribution': distance_stats,
                'avg_neighbor_distance': avg_knn_distance,
                'sample_size_analyzed': len(X_sample)
            }
            
        except Exception as e:
            return self._create_error_result(f"Error en separability analysis: {str(e)}")
    
    def analyze_feature_clustering_potential(self, df):
        """
        Identificar mejores características para clustering.
        
        Evalúa el poder discriminativo de cada característica
        y detecta redundancias para optimizar clustering.
        
        Args:
            df: DataFrame con características musicales
            
        Returns:
            dict: Ranking de características y recomendaciones
        """
        try:
            # Preparar datos
            available_features = [f for f in self.musical_features if f in df.columns]
            if not available_features:
                return self._create_error_result("No se encontraron características musicales válidas")
            
            X = df[available_features].copy()
            
            # Limpiar datos
            X = X.dropna()
            if len(X) < 10:
                return self._create_error_result("Dataset muy pequeño para análisis")
            
            # Normalizar para análisis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calcular varianza de cada característica
            feature_variances = np.var(X_scaled, axis=0)
            
            # Calcular correlaciones para detectar redundancia
            correlation_matrix = np.corrcoef(X_scaled.T)
            
            # Ranking de características por poder discriminativo
            feature_scores = []
            for i, feature in enumerate(available_features):
                # Score basado en varianza (características más diversas = mejor)
                variance_score = feature_variances[i]
                
                # Penalizar características altamente correlacionadas
                correlations = np.abs(correlation_matrix[i])
                correlations[i] = 0  # Excluir correlación consigo mismo
                max_correlation = np.max(correlations)
                redundancy_penalty = max_correlation ** 2
                
                # Score final
                final_score = variance_score * (1 - redundancy_penalty)
                
                feature_scores.append({
                    'feature': feature,
                    'variance_score': variance_score,
                    'max_correlation': max_correlation,
                    'final_score': final_score,
                    'redundancy_level': 'high' if max_correlation > 0.8 else 
                                      'medium' if max_correlation > 0.6 else 'low'
                })
            
            # Ordenar por score
            feature_scores.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Detectar características redundantes
            redundant_features = [f['feature'] for f in feature_scores if f['max_correlation'] > 0.8]
            
            # Recomendar top características
            recommended_features = [f['feature'] for f in feature_scores[:8]]  # Top 8
            
            return {
                'feature_ranking': feature_scores,
                'redundant_features': redundant_features,
                'recommended_features': recommended_features,
                'preprocessing_needed': {
                    'standardization': 'required',
                    'outlier_handling': 'recommended',
                    'feature_selection': len(recommended_features) < len(available_features)
                },
                'n_features_analyzed': len(available_features)
            }
            
        except Exception as e:
            return self._create_error_result(f"Error en feature analysis: {str(e)}")
    
    def calculate_clustering_readiness_score(self, df):
        """
        Score general de qué tan listo está el dataset para clustering.
        
        Combina múltiples métricas en un score 0-100 que indica
        la aptitud del dataset para clustering efectivo.
        
        Args:
            df: DataFrame con características musicales
            
        Returns:
            dict: Clustering readiness score y recomendaciones
        """
        try:
            # Obtener análisis individuales
            tendency = self.assess_clustering_tendency(df)
            separability = self.analyze_cluster_separability(df)
            features = self.analyze_feature_clustering_potential(df)
            
            # Verificar errores
            if any(result.get('error') for result in [tendency, separability, features]):
                return self._create_error_result("Error en uno o más análisis componentes")
            
            # Calcular componentes del score
            score_components = {}
            
            # 1. Hopkins Statistic (30 puntos)
            hopkins_score = min(30, tendency['hopkins_statistic'] * 30)
            score_components['clustering_tendency'] = hopkins_score
            
            # 2. Feature quality and diversity (25 puntos)
            n_good_features = len(features['recommended_features'])
            feature_score = min(25, (n_good_features / 8) * 25)
            score_components['feature_quality'] = feature_score
            
            # 3. Separability potential (20 puntos)
            separability_score = min(20, separability['separability_score'] * 20)
            score_components['separability'] = separability_score
            
            # 4. Distribution compatibility (15 puntos)
            # Basado en número de características con alta varianza
            high_variance_features = sum(1 for f in features['feature_ranking'] if f['variance_score'] > 0.5)
            distribution_score = min(15, (high_variance_features / len(features['feature_ranking'])) * 15)
            score_components['distribution_compatibility'] = distribution_score
            
            # 5. Preprocessing complexity (10 puntos) - menos complejidad = más puntos
            redundant_ratio = len(features['redundant_features']) / len(features['feature_ranking'])
            preprocessing_score = max(0, 10 * (1 - redundant_ratio))
            score_components['preprocessing_simplicity'] = preprocessing_score
            
            # Score total
            total_score = sum(score_components.values())
            
            # Determinar nivel de preparación
            if total_score >= 80:
                readiness_level = 'excellent'
            elif total_score >= 60:
                readiness_level = 'good'
            elif total_score >= 40:
                readiness_level = 'fair'
            else:
                readiness_level = 'poor'
            
            # Generar recomendaciones de mejora
            improvements = self._generate_improvement_suggestions(
                score_components, tendency, separability, features
            )
            
            return {
                'readiness_score': round(total_score, 1),
                'readiness_level': readiness_level,
                'score_breakdown': score_components,
                'improvement_suggestions': improvements,
                'component_analysis': {
                    'clustering_tendency': tendency,
                    'separability_analysis': separability,
                    'feature_analysis': features
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Error calculando readiness score: {str(e)}")
    
    # === MÉTODOS AUXILIARES ===
    
    def _prepare_data(self, df):
        """Preparar datos para análisis de clustering."""
        try:
            # Seleccionar características disponibles
            available_features = [f for f in self.musical_features if f in df.columns]
            if not available_features:
                return None
            
            # Extraer y limpiar datos
            X = df[available_features].copy()
            X = X.dropna()
            
            if len(X) < 10:
                return None
            
            # Normalizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled
            
        except Exception:
            return None
    
    def _calculate_hopkins_statistic(self, X, sample_size=None):
        """Calcular Hopkins Statistic para clustering tendency."""
        try:
            n, d = X.shape
            if sample_size is None:
                sample_size = min(int(0.1 * n), 200)  # 10% o máximo 200 puntos
            
            # Generar puntos aleatorios uniformes en el espacio de datos
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            uniform_points = np.random.uniform(min_vals, max_vals, size=(sample_size, d))
            
            # Seleccionar muestra aleatoria de datos reales
            sample_indices = np.random.choice(n, sample_size, replace=False)
            real_sample = X[sample_indices]
            
            # Calcular distancias mínimas para puntos uniformes (U_i)
            nbrs_uniform = NearestNeighbors(n_neighbors=1).fit(X)
            u_distances, _ = nbrs_uniform.kneighbors(uniform_points)
            U = np.sum(u_distances)
            
            # Calcular distancias mínimas para puntos reales (W_i)
            remaining_data = np.delete(X, sample_indices, axis=0)
            nbrs_real = NearestNeighbors(n_neighbors=1).fit(remaining_data)
            w_distances, _ = nbrs_real.kneighbors(real_sample)
            W = np.sum(w_distances)
            
            # Hopkins Statistic
            hopkins = U / (U + W)
            
            return float(hopkins)
            
        except Exception:
            return 0.5  # Valor neutral en caso de error
    
    def _find_optimal_k(self, metrics):
        """Determinar K óptimo basado en múltiples métricas."""
        try:
            k_values = metrics['k_values']
            
            # Método del codo para inercia
            inertias = metrics['inertias']
            elbow_k = self._find_elbow_point(k_values, inertias)
            
            # Máximo Silhouette Score
            silhouette_scores = metrics['silhouette_scores']
            valid_silhouettes = [(k, s) for k, s in zip(k_values, silhouette_scores) if s > -1]
            
            if valid_silhouettes:
                silhouette_k = max(valid_silhouettes, key=lambda x: x[1])[0]
                max_silhouette = max(valid_silhouettes, key=lambda x: x[1])[1]
            else:
                silhouette_k = k_values[len(k_values)//2]  # K medio como fallback
                max_silhouette = -1
            
            # Máximo Calinski-Harabasz
            calinski_scores = metrics['calinski_harabasz_scores']
            calinski_k = k_values[np.argmax(calinski_scores)]
            
            # Combinar recomendaciones
            recommendations = [elbow_k, silhouette_k, calinski_k]
            recommendations = [k for k in recommendations if k is not None]
            
            if recommendations:
                # K más frecuente o promedio
                from collections import Counter
                k_counts = Counter(recommendations)
                if len(k_counts) == len(recommendations):  # Todos diferentes
                    best_k = int(np.mean(recommendations))
                else:
                    best_k = k_counts.most_common(1)[0][0]
                
                # Calcular acuerdo entre métodos
                agreement = k_counts.most_common(1)[0][1] / len(recommendations)
            else:
                best_k = k_values[len(k_values)//2]  # K medio
                agreement = 0.0
            
            return {
                'best_k': best_k,
                'k_range': (max(2, best_k-1), min(len(k_values)+1, best_k+2)),
                'agreement': agreement,
                'quality': {
                    'expected_silhouette': max_silhouette if max_silhouette > -1 else 'unknown',
                    'elbow_k': elbow_k,
                    'silhouette_k': silhouette_k,
                    'calinski_k': calinski_k
                },
                'notes': f"Métodos usados: Elbow, Silhouette, Calinski-Harabasz"
            }
            
        except Exception:
            return {
                'best_k': 4,  # Default
                'k_range': (3, 6),
                'agreement': 0.0,
                'quality': {'expected_silhouette': 'error'},
                'notes': "Error en análisis - usando valores por defecto"
            }
    
    def _find_elbow_point(self, k_values, inertias):
        """Encontrar punto de codo en curva de inercia."""
        try:
            if len(inertias) < 3:
                return None
                
            # Calcular diferencias entre puntos consecutivos
            diffs = np.diff(inertias)
            diff_ratios = np.diff(diffs)
            
            # Encontrar el punto donde la mejora se desacelera más
            if len(diff_ratios) > 0:
                elbow_idx = np.argmax(diff_ratios) + 2  # +2 por los dos diff()
                if elbow_idx < len(k_values):
                    return k_values[elbow_idx]
            
            return None
            
        except Exception:
            return None
    
    def _calculate_separability_score(self, distance_stats, avg_knn_distance):
        """Calcular score de separabilidad basado en distribución de distancias."""
        try:
            # Relación entre distancia promedio y variabilidad
            mean_dist = distance_stats['mean_distance']
            std_dist = distance_stats['std_distance']
            
            if mean_dist == 0:
                return 0.0
            
            # Coeficiente de variación (menor = más homogéneo = peor separabilidad)
            cv = std_dist / mean_dist
            
            # Relación entre distancia media global y distancia a vecinos
            neighbor_ratio = avg_knn_distance / mean_dist if mean_dist > 0 else 0
            
            # Score combinado (0-1)
            separability = min(1.0, (cv + neighbor_ratio) / 2)
            
            return float(separability)
            
        except Exception:
            return 0.5
    
    def _predict_silhouette_range(self, separability_score, distance_stats):
        """Predecir rango esperado de Silhouette Score."""
        try:
            # Basado en separabilidad y distribución de distancias
            base_silhouette = separability_score * 0.6  # Max teórico ~0.6
            
            # Ajustar por variabilidad de distancias
            cv = distance_stats['std_distance'] / distance_stats['mean_distance']
            variability_factor = min(1.2, max(0.8, cv))
            
            predicted_silhouette = base_silhouette * variability_factor
            
            # Rango con incertidumbre
            uncertainty = 0.1
            lower_bound = max(0.0, predicted_silhouette - uncertainty)
            upper_bound = min(1.0, predicted_silhouette + uncertainty)
            
            return (round(lower_bound, 3), round(upper_bound, 3))
            
        except Exception:
            return (0.0, 0.3)
    
    def _assess_overlap_risk(self, distances, avg_knn_distance):
        """Evaluar riesgo de overlap entre clusters."""
        try:
            # Si la distancia promedio a vecinos es muy similar a la distancia global
            global_mean = np.mean(distances)
            
            if global_mean == 0:
                return "high"
            
            ratio = avg_knn_distance / global_mean
            
            if ratio > 0.8:
                return "high - datos muy densos"
            elif ratio > 0.6:
                return "medium - separabilidad moderada"
            else:
                return "low - buena separabilidad esperada"
                
        except Exception:
            return "unknown"
    
    def _generate_improvement_suggestions(self, score_components, tendency, separability, features):
        """Generar sugerencias de mejora basadas en análisis."""
        suggestions = []
        
        # Sugerencias basadas en Hopkins Statistic
        if tendency['hopkins_statistic'] < 0.5:
            suggestions.append("CRÍTICO: Datos tienden a ser aleatorios. Considerar aumentar diversidad o usar muestreo estratificado.")
        elif tendency['hopkins_statistic'] < 0.6:
            suggestions.append("Mejorar clustering tendency con selección más diversa de características.")
        
        # Sugerencias basadas en separabilidad
        if separability['separability_score'] < 0.3:
            suggestions.append("PROBLEMA: Baja separabilidad. Aplicar feature engineering o usar algoritmos como DBSCAN.")
        elif separability['separability_score'] < 0.5:
            suggestions.append("Separabilidad mejorable. Considerar transformaciones logarítmicas o PCA.")
        
        # Sugerencias basadas en características
        if len(features['redundant_features']) > 2:
            suggestions.append(f"Eliminar características redundantes: {', '.join(features['redundant_features'][:3])}")
        
        if len(features['recommended_features']) < 6:
            suggestions.append("Pocas características útiles. Considerar feature engineering para crear características sintéticas.")
        
        # Sugerencias generales
        if score_components['clustering_tendency'] < 15:
            suggestions.append("Usar pipeline de selección con mayor diversidad musical.")
        
        if not suggestions:
            suggestions.append("Dataset en buen estado para clustering. Proceder con K-Means estándar.")
        
        return suggestions
    
    def _create_error_result(self, error_message):
        """Crear resultado de error estandarizado."""
        return {
            'error': True,
            'message': error_message,
            'timestamp': pd.Timestamp.now().isoformat()
        }