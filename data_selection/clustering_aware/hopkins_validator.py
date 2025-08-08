#!/usr/bin/env python3
"""
Hopkins Validator - Sistema de Validación Hopkins Statistic Integrado

Este módulo proporciona validación continua de clustering tendency durante
el proceso de selección de datos, permitiendo feedback en tiempo real y
estrategias de recuperación cuando la clusterabilidad se degrada.

Funcionalidades:
- Cálculo eficiente Hopkins Statistic optimizado para feedback continuo
- Validación con thresholds configurables
- Historial de validaciones para análisis posterior
- Estrategias de fallback para casos problemáticos

Uso típico:
    validator = HopkinsValidator(threshold=0.70)
    result = validator.validate_during_selection(current_data)
    if result['action'] == 'fallback':
        # Aplicar estrategia alternativa
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

warnings.filterwarnings('ignore')

class HopkinsValidator:
    """
    Validador Hopkins Statistic integrado en proceso de selección.
    
    Diseñado para proporcionar feedback continuo durante la selección de datos,
    permitiendo detectar degradación de clustering tendency y aplicar medidas
    correctivas en tiempo real.
    """
    
    def __init__(self, threshold: float = 0.70, sample_size_ratio: float = 0.05):
        """
        Inicializar validador Hopkins.
        
        Args:
            threshold: Threshold mínimo Hopkins para considerarlo aceptable
            sample_size_ratio: Ratio del dataset para muestreo (velocidad vs precisión)
        """
        self.threshold = threshold
        self.sample_size_ratio = sample_size_ratio
        self.validation_history: List[Dict] = []
        self.min_sample_size = 10
        self.max_sample_size = 200
        
    def calculate_hopkins_fast(self, data_sample: Union[np.ndarray, pd.DataFrame], 
                              sample_size: Optional[int] = None) -> float:
        """
        Cálculo eficiente Hopkins Statistic optimizado para feedback continuo.
        
        Implementación optimizada que balance precisión con velocidad para
        permitir validación durante proceso de selección sin impacto significativo
        en performance.
        
        Args:
            data_sample: Datos a analizar (numpy array o DataFrame)
            sample_size: Tamaño muestra para cálculo (None = automático)
            
        Returns:
            float: Hopkins Statistic (0-1, >0.5 indica clustering tendency)
        """
        try:
            # Convertir a numpy array si es necesario
            if isinstance(data_sample, pd.DataFrame):
                data_array = data_sample.values
            else:
                data_array = np.asarray(data_sample)
            
            n, d = data_array.shape
            
            # Validaciones básicas
            if n < self.min_sample_size:
                return 0.5  # Datos insuficientes, valor neutro
                
            if d == 0:
                return 0.5  # Sin características, valor neutro
            
            # Determinar tamaño de muestra optimal
            if sample_size is None:
                sample_size = min(
                    self.max_sample_size,
                    max(self.min_sample_size, int(n * self.sample_size_ratio))
                )
            
            sample_size = min(sample_size, n // 2)  # No exceder 50% de datos
            
            if sample_size < self.min_sample_size:
                return 0.5  # Muestra muy pequeña
            
            # Generar puntos uniformes aleatorios en el espacio de datos
            min_vals = np.min(data_array, axis=0)
            max_vals = np.max(data_array, axis=0)
            
            # Verificar que hay variación en los datos
            if np.all(min_vals == max_vals):
                return 0.0  # Datos constantes = no clusterizable
            
            uniform_points = np.random.uniform(min_vals, max_vals, size=(sample_size, d))
            
            # Seleccionar muestra aleatoria de datos reales
            if n > sample_size:
                sample_indices = np.random.choice(n, sample_size, replace=False)
                real_sample = data_array[sample_indices]
            else:
                sample_indices = np.arange(n)
                real_sample = data_array
                sample_size = n
            
            # U_i: distancias de puntos uniformes a datos reales (nearest neighbor)
            nbrs_uniform = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data_array)
            u_distances, _ = nbrs_uniform.kneighbors(uniform_points)
            U = np.sum(u_distances)
            
            # W_i: distancias de puntos reales a otros puntos reales
            # Usar datos restantes (no incluidos en muestra) como referencia
            remaining_indices = np.setdiff1d(np.arange(n), sample_indices)
            
            if len(remaining_indices) > 0:
                remaining_data = data_array[remaining_indices]
                nbrs_real = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(remaining_data)
                w_distances, _ = nbrs_real.kneighbors(real_sample)
                W = np.sum(w_distances)
            else:
                # Fallback: usar todos los datos con k=2 (excluir punto mismo)
                if n > 1:
                    nbrs_real = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data_array)
                    w_distances, _ = nbrs_real.kneighbors(real_sample)
                    # Usar segunda distancia (la primera es distancia 0 al mismo punto)
                    W = np.sum(w_distances[:, 1])
                else:
                    W = U  # Solo un punto, usar U como fallback
            
            # Hopkins Statistic
            denominator = U + W
            if denominator > 0:
                hopkins = U / denominator
            else:
                hopkins = 0.5  # Fallback neutral
            
            return float(np.clip(hopkins, 0.0, 1.0))  # Asegurar rango [0,1]
            
        except Exception as e:
            # En caso de error, retornar valor neutro y loguear
            print(f"⚠️  Error en Hopkins calculation: {e}")
            return 0.5  # Valor neutro conservativo
    
    def validate_during_selection(self, current_selection: Union[np.ndarray, pd.DataFrame], 
                                 iteration: Optional[Union[int, str]] = None,
                                 context: Optional[str] = None) -> Dict:
        """
        Validar Hopkins durante proceso de selección con recomendación de acción.
        
        Este método es el core del sistema de validación continua. Calcula Hopkins
        de la selección actual y determina si debe continuar o aplicar fallback.
        
        Args:
            current_selection: Datos seleccionados hasta el momento
            iteration: Número de iteración o identificador del paso
            context: Contexto adicional para logging
            
        Returns:
            dict: {
                'action': 'continue' | 'fallback',
                'hopkins': float,
                'recommendation': str,
                'details': dict
            }
        """
        validation_timestamp = datetime.now()
        
        # Validaciones iniciales
        if isinstance(current_selection, pd.DataFrame):
            if len(current_selection) == 0:
                return self._create_validation_result(
                    action='continue',
                    hopkins=None,
                    recommendation='Dataset vacío - continuar selección',
                    timestamp=validation_timestamp,
                    iteration=iteration,
                    context=context
                )
            data_for_analysis = current_selection.select_dtypes(include=[np.number])
        else:
            data_for_analysis = current_selection
        
        if len(data_for_analysis) < self.min_sample_size:
            return self._create_validation_result(
                action='continue',
                hopkins=None,
                recommendation=f'Datos insuficientes ({len(data_for_analysis)} < {self.min_sample_size}) - continuar',
                timestamp=validation_timestamp,
                iteration=iteration,
                context=context
            )
        
        # Calcular Hopkins
        hopkins_score = self.calculate_hopkins_fast(data_for_analysis)
        
        # Determinar acción basada en threshold
        if hopkins_score >= self.threshold:
            action = 'continue'
            recommendation = f'Hopkins {hopkins_score:.3f} >= {self.threshold} - Selección adecuada'
            status = 'good'
        elif hopkins_score >= (self.threshold - 0.10):  # Grace margin
            action = 'continue'
            recommendation = f'Hopkins {hopkins_score:.3f} en zona de gracia - Monitorear'
            status = 'warning'
        else:
            action = 'fallback'
            recommendation = f'Hopkins {hopkins_score:.3f} < {self.threshold} - Activar estrategia fallback'
            status = 'critical'
        
        # Crear registro de validación
        validation_result = self._create_validation_result(
            action=action,
            hopkins=hopkins_score,
            recommendation=recommendation,
            timestamp=validation_timestamp,
            iteration=iteration,
            context=context,
            status=status
        )
        
        # Guardar en historial
        self.validation_history.append(validation_result.copy())
        
        return validation_result
    
    def _create_validation_result(self, action: str, hopkins: Optional[float], 
                                recommendation: str, timestamp: datetime,
                                iteration: Optional[Union[int, str]] = None,
                                context: Optional[str] = None,
                                status: str = 'unknown') -> Dict:
        """Crear resultado estructurado de validación."""
        
        return {
            'action': action,
            'hopkins': hopkins,
            'recommendation': recommendation,
            'details': {
                'timestamp': timestamp.isoformat(),
                'iteration': iteration,
                'context': context,
                'status': status,
                'threshold_used': self.threshold,
                'sample_size': len(self.validation_history) if hasattr(self, 'validation_history') else 0
            }
        }
    
    def get_validation_summary(self) -> Dict:
        """
        Obtener resumen comprehensivo de todas las validaciones realizadas.
        
        Returns:
            dict: Estadísticas y análisis del historial de validaciones
        """
        if not self.validation_history:
            return {
                'status': 'No validations performed',
                'total_validations': 0
            }
        
        # Extraer Hopkins scores válidos
        hopkins_scores = [
            v['hopkins'] for v in self.validation_history 
            if v['hopkins'] is not None
        ]
        
        # Contar acciones
        actions = [v['action'] for v in self.validation_history]
        action_counts = {
            'continue': actions.count('continue'),
            'fallback': actions.count('fallback')
        }
        
        # Contar status
        statuses = [v['details'].get('status', 'unknown') for v in self.validation_history]
        status_counts = {
            'good': statuses.count('good'),
            'warning': statuses.count('warning'),
            'critical': statuses.count('critical'),
            'unknown': statuses.count('unknown')
        }
        
        # Análisis temporal
        if len(hopkins_scores) >= 2:
            hopkins_trend = 'improving' if hopkins_scores[-1] > hopkins_scores[0] else \
                           'degrading' if hopkins_scores[-1] < hopkins_scores[0] else 'stable'
        else:
            hopkins_trend = 'insufficient_data'
        
        # Calcular estadísticas
        summary = {
            'total_validations': len(self.validation_history),
            'successful_validations': len(hopkins_scores),
            'hopkins_statistics': {
                'scores': hopkins_scores,
                'avg_hopkins': np.mean(hopkins_scores) if hopkins_scores else None,
                'min_hopkins': np.min(hopkins_scores) if hopkins_scores else None,
                'max_hopkins': np.max(hopkins_scores) if hopkins_scores else None,
                'std_hopkins': np.std(hopkins_scores) if len(hopkins_scores) > 1 else None,
                'trend': hopkins_trend
            },
            'action_summary': action_counts,
            'status_summary': status_counts,
            'quality_assessment': {
                'threshold_used': self.threshold,
                'threshold_violations': action_counts['fallback'],
                'success_rate': action_counts['continue'] / len(self.validation_history) if self.validation_history else 0,
                'overall_quality': self._assess_overall_quality(hopkins_scores, action_counts)
            },
            'timestamps': {
                'first_validation': self.validation_history[0]['details']['timestamp'],
                'last_validation': self.validation_history[-1]['details']['timestamp']
            }
        }
        
        return summary
    
    def _assess_overall_quality(self, hopkins_scores: List[float], 
                               action_counts: Dict[str, int]) -> str:
        """Evaluar calidad general del proceso de validación."""
        
        if not hopkins_scores:
            return 'no_data'
        
        avg_hopkins = np.mean(hopkins_scores)
        success_rate = action_counts['continue'] / (action_counts['continue'] + action_counts['fallback'])
        
        if avg_hopkins >= 0.80 and success_rate >= 0.90:
            return 'excellent'
        elif avg_hopkins >= 0.70 and success_rate >= 0.80:
            return 'good' 
        elif avg_hopkins >= 0.60 and success_rate >= 0.70:
            return 'acceptable'
        elif avg_hopkins >= 0.50 and success_rate >= 0.60:
            return 'marginal'
        else:
            return 'poor'
    
    def reset_validation_history(self):
        """Limpiar historial de validaciones para nueva sesión."""
        self.validation_history = []
    
    def export_validation_log(self, filepath: Optional[str] = None) -> str:
        """
        Exportar log de validaciones a archivo JSON.
        
        Args:
            filepath: Path para guardar (None = generar automáticamente)
            
        Returns:
            str: Path del archivo generado
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"hopkins_validation_log_{timestamp}.json"
        
        import json
        
        export_data = {
            'validation_summary': self.get_validation_summary(),
            'detailed_history': self.validation_history,
            'configuration': {
                'threshold': self.threshold,
                'sample_size_ratio': self.sample_size_ratio,
                'min_sample_size': self.min_sample_size,
                'max_sample_size': self.max_sample_size
            },
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_validations': len(self.validation_history)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def compare_with_baseline(self, baseline_hopkins: float) -> Dict:
        """
        Comparar performance actual con Hopkins baseline.
        
        Args:
            baseline_hopkins: Hopkins del dataset original/baseline
            
        Returns:
            dict: Análisis comparativo detallado
        """
        summary = self.get_validation_summary()
        
        if not summary.get('hopkins_statistics', {}).get('scores'):
            return {
                'status': 'no_data_for_comparison',
                'baseline_hopkins': baseline_hopkins
            }
        
        current_avg = summary['hopkins_statistics']['avg_hopkins']
        preservation_ratio = current_avg / baseline_hopkins if baseline_hopkins > 0 else 1
        
        # Evaluación de preservación
        if preservation_ratio >= 0.90:
            preservation_level = 'excellent'
        elif preservation_ratio >= 0.80:
            preservation_level = 'good'
        elif preservation_ratio >= 0.70:
            preservation_level = 'acceptable'
        elif preservation_ratio >= 0.60:
            preservation_level = 'marginal'
        else:
            preservation_level = 'poor'
        
        return {
            'baseline_hopkins': baseline_hopkins,
            'current_avg_hopkins': current_avg,
            'preservation_ratio': preservation_ratio,
            'preservation_level': preservation_level,
            'absolute_difference': current_avg - baseline_hopkins,
            'relative_change_percent': (preservation_ratio - 1) * 100,
            'recommendation': self._generate_preservation_recommendation(preservation_level, preservation_ratio)
        }
    
    def _generate_preservation_recommendation(self, level: str, ratio: float) -> str:
        """Generar recomendación basada en nivel de preservación."""
        
        recommendations = {
            'excellent': f"Hopkins preservation excelente ({ratio:.1%}). Continuar con selección actual.",
            'good': f"Buena preservación Hopkins ({ratio:.1%}). Monitorear para mantener calidad.",
            'acceptable': f"Preservación aceptable ({ratio:.1%}). Considerar ajustes menores al algoritmo.",
            'marginal': f"Preservación marginal ({ratio:.1%}). Revisar estrategia de selección o aplicar fallback.",
            'poor': f"Preservación deficiente ({ratio:.1%}). Cambio crítico de estrategia requerido."
        }
        
        return recommendations.get(level, "Nivel de preservación desconocido - revisar implementación.")


# Funciones de utilidad adicionales
def quick_hopkins_check(data: Union[np.ndarray, pd.DataFrame], 
                       threshold: float = 0.70) -> Dict:
    """
    Check rápido de Hopkins para validación instantánea.
    
    Args:
        data: Datos a analizar
        threshold: Threshold para evaluación
        
    Returns:
        dict: Resultado rápido con recomendación
    """
    validator = HopkinsValidator(threshold=threshold)
    hopkins = validator.calculate_hopkins_fast(data)
    
    return {
        'hopkins_statistic': hopkins,
        'is_clusterable': hopkins > 0.5,
        'meets_threshold': hopkins >= threshold,
        'quality_level': 'excellent' if hopkins >= 0.80 else
                        'good' if hopkins >= 0.70 else
                        'acceptable' if hopkins >= 0.60 else
                        'marginal' if hopkins >= 0.50 else 'poor',
        'recommendation': 'Proceed with clustering' if hopkins >= threshold else
                         'Consider data preprocessing or alternative methods'
    }


def compare_datasets_hopkins(dataset1: Union[np.ndarray, pd.DataFrame],
                           dataset2: Union[np.ndarray, pd.DataFrame],
                           labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> Dict:
    """
    Comparar Hopkins Statistic entre dos datasets.
    
    Args:
        dataset1, dataset2: Datasets a comparar
        labels: Etiquetas descriptivas para los datasets
        
    Returns:
        dict: Comparación detallada
    """
    validator = HopkinsValidator()
    
    hopkins1 = validator.calculate_hopkins_fast(dataset1)
    hopkins2 = validator.calculate_hopkins_fast(dataset2)
    
    difference = hopkins2 - hopkins1
    relative_change = (difference / hopkins1 * 100) if hopkins1 > 0 else 0
    
    better_dataset = labels[1] if hopkins2 > hopkins1 else labels[0]
    
    return {
        'comparison': {
            labels[0]: {
                'hopkins': hopkins1,
                'clusterable': hopkins1 > 0.5,
                'quality': quick_hopkins_check(dataset1)['quality_level']
            },
            labels[1]: {
                'hopkins': hopkins2,
                'clusterable': hopkins2 > 0.5,
                'quality': quick_hopkins_check(dataset2)['quality_level']
            }
        },
        'difference_analysis': {
            'absolute_difference': difference,
            'relative_change_percent': relative_change,
            'better_dataset': better_dataset,
            'significant_difference': abs(difference) > 0.05
        },
        'recommendation': f"{better_dataset} shows better clustering tendency" if abs(difference) > 0.05 
                         else "Both datasets show similar clustering tendency"
    }


if __name__ == "__main__":
    # Ejemplo de uso y tests básicos
    print("🧪 Hopkins Validator - Tests básicos")
    
    # Test con datos sintéticos
    np.random.seed(42)
    
    # Datos clusterizables
    cluster1 = np.random.normal([0, 0], 0.5, (100, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (100, 2))
    clusterable_data = np.vstack([cluster1, cluster2])
    
    # Datos aleatorios
    random_data = np.random.uniform(-4, 4, (200, 2))
    
    # Validador
    validator = HopkinsValidator(threshold=0.70)
    
    # Test datos clusterizables
    result_clusterable = validator.validate_during_selection(
        clusterable_data, iteration="test_clusterable"
    )
    print(f"Datos clusterizables: Hopkins={result_clusterable['hopkins']:.3f}, Action={result_clusterable['action']}")
    
    # Test datos aleatorios
    result_random = validator.validate_during_selection(
        random_data, iteration="test_random"
    )
    print(f"Datos aleatorios: Hopkins={result_random['hopkins']:.3f}, Action={result_random['action']}")
    
    # Resumen
    summary = validator.get_validation_summary()
    print(f"\nResumen: {summary['total_validations']} validaciones, calidad {summary['quality_assessment']['overall_quality']}")
    
    # Comparación directa
    comparison = compare_datasets_hopkins(clusterable_data, random_data, 
                                        ('Clusterable', 'Random'))
    print(f"Comparación: {comparison['difference_analysis']['better_dataset']} es mejor")
    
    print("✅ Tests básicos completados")