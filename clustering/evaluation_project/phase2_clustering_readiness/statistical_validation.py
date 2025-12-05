#!/usr/bin/env python3
"""
Statistical Validation Module - FASE 2
======================================

Módulo especializado para validación estadística rigurosa de resultados
de clustering readiness assessment comparativo. Implementa pruebas de
significancia, análisis de tamaño de efecto, y validación de hipótesis
para sustentar conclusiones académicas sólidas.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import pandas as pd
from sklearn.utils import resample

class StatisticalValidator:
    """
    Validador estadístico especializado para análisis de clustering readiness
    que implementa metodología científica rigurosa para validación de hipótesis.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializar validador estadístico.
        
        Args:
            logger: Logger configurado para trazabilidad
        """
        self.logger = logger
        self.alpha_levels = {
            'strict': 0.01,      # Nivel estricto para conclusiones definitivas
            'standard': 0.05,    # Nivel estándar para investigación
            'exploratory': 0.1   # Nivel exploratorio para tendencias
        }
        
        self.logger.info("StatisticalValidator inicializado")
        self.logger.info(f"Niveles alpha configurados: {self.alpha_levels}")
    
    def validate_comparative_significance(self, hopkins_results: Dict[str, Any],
                                        dimensionality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar validación estadística comprehensive de resultados comparativos.
        
        Args:
            hopkins_results: Resultados análisis Hopkins comparativo
            dimensionality_results: Resultados análisis dimensional
            
        Returns:
            Validación estadística completa con pruebas de significancia
        """
        self.logger.info("INICIANDO VALIDACIÓN ESTADÍSTICA COMPARATIVA")
        
        validation_results = {
            'hopkins_significance_analysis': {},
            'effect_size_analysis': {},
            'confidence_intervals': {},
            'bootstrap_validation': {},
            'normality_tests': {},
            'power_analysis': {},
            'multiple_comparisons_adjustment': {},
            'conclusion_validity': {}
        }
        
        # 1. Análisis de significancia Hopkins
        self.logger.info("Ejecutando análisis significancia Hopkins")
        validation_results['hopkins_significance_analysis'] = self._analyze_hopkins_significance(hopkins_results)
        
        # 2. Análisis tamaño de efecto
        self.logger.info("Calculando análisis tamaño de efecto")
        validation_results['effect_size_analysis'] = self._calculate_effect_sizes(hopkins_results)
        
        # 3. Intervalos de confianza
        self.logger.info("Generando intervalos de confianza")
        validation_results['confidence_intervals'] = self._calculate_confidence_intervals(hopkins_results)
        
        # 4. Validación bootstrap
        self.logger.info("Ejecutando validación bootstrap")
        validation_results['bootstrap_validation'] = self._perform_bootstrap_validation(hopkins_results)
        
        # 5. Tests de normalidad
        self.logger.info("Ejecutando tests de normalidad")
        validation_results['normality_tests'] = self._perform_normality_tests(hopkins_results)
        
        # 6. Análisis de poder estadístico
        self.logger.info("Calculando análisis de poder estadístico")
        validation_results['power_analysis'] = self._calculate_statistical_power(hopkins_results)
        
        # 7. Ajuste comparaciones múltiples
        self.logger.info("Aplicando ajuste comparaciones múltiples")
        validation_results['multiple_comparisons_adjustment'] = self._adjust_multiple_comparisons(hopkins_results)
        
        # 8. Validez de conclusiones
        self.logger.info("Evaluando validez de conclusiones")
        validation_results['conclusion_validity'] = self._evaluate_conclusion_validity(validation_results)
        
        self.logger.info("VALIDACIÓN ESTADÍSTICA COMPLETADA")
        
        return validation_results
    
    def _analyze_hopkins_significance(self, hopkins_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar significancia estadística de diferencias Hopkins.
        
        Args:
            hopkins_results: Resultados Hopkins comparativos
            
        Returns:
            Análisis completo de significancia estadística
        """
        significance_analysis = {}
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            self.logger.warning("Datos insuficientes para análisis de significancia")
            return significance_analysis
        
        # Extraer valores de iteraciones
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        
        if not (semantic_values and musical_values):
            self.logger.warning("No se encontraron valores de iteración")
            return significance_analysis
        
        # Test t pareado (más apropiado que independiente)
        try:
            t_stat_paired, p_value_paired = stats.ttest_rel(musical_values, semantic_values)
            
            significance_analysis['paired_t_test'] = {
                't_statistic': float(t_stat_paired),
                'p_value': float(p_value_paired),
                'degrees_freedom': len(musical_values) - 1,
                'significant_strict': p_value_paired < self.alpha_levels['strict'],
                'significant_standard': p_value_paired < self.alpha_levels['standard'],
                'significant_exploratory': p_value_paired < self.alpha_levels['exploratory']
            }
            
            self.logger.info(f"Test t pareado: t={t_stat_paired:.4f}, p={p_value_paired:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error en test t pareado: {e}")
        
        # Test Wilcoxon signed-rank (alternativa no paramétrica)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(musical_values, semantic_values)
            
            significance_analysis['wilcoxon_test'] = {
                'statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant_strict': wilcoxon_p < self.alpha_levels['strict'],
                'significant_standard': wilcoxon_p < self.alpha_levels['standard'],
                'significant_exploratory': wilcoxon_p < self.alpha_levels['exploratory']
            }
            
            self.logger.info(f"Test Wilcoxon: W={wilcoxon_stat:.4f}, p={wilcoxon_p:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error en test Wilcoxon: {e}")
        
        # Test Mann-Whitney U (independiente no paramétrico)
        try:
            u_stat, u_p = stats.mannwhitneyu(musical_values, semantic_values, alternative='two-sided')
            
            significance_analysis['mann_whitney_test'] = {
                'u_statistic': float(u_stat),
                'p_value': float(u_p),
                'significant_strict': u_p < self.alpha_levels['strict'],
                'significant_standard': u_p < self.alpha_levels['standard'],
                'significant_exploratory': u_p < self.alpha_levels['exploratory']
            }
            
        except Exception as e:
            self.logger.error(f"Error en test Mann-Whitney: {e}")
        
        return significance_analysis
    
    def _calculate_effect_sizes(self, hopkins_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular múltiples medidas de tamaño de efecto.
        
        Args:
            hopkins_results: Resultados Hopkins
            
        Returns:
            Análisis completo de tamaños de efecto
        """
        effect_analysis = {}
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            return effect_analysis
        
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        
        if not (semantic_values and musical_values):
            return effect_analysis
        
        # Cohen's d para muestras pareadas
        differences = np.array(musical_values) - np.array(semantic_values)
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Cohen's d para muestras independientes (referencia)
        pooled_std = np.sqrt((np.var(semantic_values, ddof=1) + np.var(musical_values, ddof=1)) / 2)
        cohens_d_independent = (np.mean(musical_values) - np.mean(semantic_values)) / pooled_std
        
        # Hedges' g (corrección para muestras pequeñas)
        n = len(musical_values)
        correction_factor = 1 - (3 / (4 * (n - 1) - 1))
        hedges_g = cohens_d * correction_factor
        
        # Glass's delta
        glass_delta = (np.mean(musical_values) - np.mean(semantic_values)) / np.std(semantic_values, ddof=1)
        
        # Cliff's delta (medida no paramétrica)
        cliff_delta = self._calculate_cliff_delta(semantic_values, musical_values)
        
        # Common Language Effect Size
        cles = self._calculate_common_language_effect_size(semantic_values, musical_values)
        
        effect_analysis = {
            'cohens_d_paired': float(cohens_d),
            'cohens_d_independent': float(cohens_d_independent),
            'hedges_g': float(hedges_g),
            'glass_delta': float(glass_delta),
            'cliff_delta': float(cliff_delta),
            'common_language_effect_size': float(cles),
            'effect_size_interpretations': {
                'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
                'cliff_delta_interpretation': self._interpret_cliff_delta(cliff_delta),
                'cles_interpretation': self._interpret_cles(cles)
            }
        }
        
        self.logger.info(f"Cohen's d (pareado): {cohens_d:.4f} ({self._interpret_cohens_d(cohens_d)})")
        self.logger.info(f"Cliff's delta: {cliff_delta:.4f} ({self._interpret_cliff_delta(cliff_delta)})")
        
        return effect_analysis
    
    def _calculate_confidence_intervals(self, hopkins_results: Dict[str, Any], 
                                      confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict[str, Any]:
        """
        Calcular intervalos de confianza para diferencias Hopkins.
        
        Args:
            hopkins_results: Resultados Hopkins
            confidence_levels: Niveles de confianza a calcular
            
        Returns:
            Intervalos de confianza para múltiples niveles
        """
        ci_analysis = {}
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            return ci_analysis
        
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        
        if not (semantic_values and musical_values):
            return ci_analysis
        
        differences = np.array(musical_values) - np.array(semantic_values)
        
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            
            # Intervalo de confianza paramétrico (t-Student)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)
            
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * (std_diff / np.sqrt(n))
            
            ci_parametric = (mean_diff - margin_error, mean_diff + margin_error)
            
            # Intervalo de confianza bootstrap
            ci_bootstrap = self._bootstrap_confidence_interval(differences, confidence_level)
            
            ci_analysis[f'confidence_{int(confidence_level*100)}'] = {
                'level': confidence_level,
                'parametric_ci': ci_parametric,
                'bootstrap_ci': ci_bootstrap,
                'mean_difference': float(mean_diff),
                'margin_error_parametric': float(margin_error),
                'contains_zero_parametric': ci_parametric[0] <= 0 <= ci_parametric[1],
                'contains_zero_bootstrap': ci_bootstrap[0] <= 0 <= ci_bootstrap[1] if ci_bootstrap else None
            }
        
        return ci_analysis
    
    def _perform_bootstrap_validation(self, hopkins_results: Dict[str, Any], 
                                    n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Ejecutar validación bootstrap para robustez de resultados.
        
        Args:
            hopkins_results: Resultados Hopkins
            n_bootstrap: Número de muestras bootstrap
            
        Returns:
            Análisis de validación bootstrap
        """
        bootstrap_analysis = {}
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            return bootstrap_analysis
        
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        
        if not (semantic_values and musical_values):
            return bootstrap_analysis
        
        self.logger.info(f"Ejecutando {n_bootstrap} muestras bootstrap")
        
        # Bootstrap de diferencias
        differences = np.array(musical_values) - np.array(semantic_values)
        bootstrap_differences = []
        bootstrap_means_semantic = []
        bootstrap_means_musical = []
        
        for i in range(n_bootstrap):
            # Resample con reemplazo
            boot_semantic = resample(semantic_values, random_state=i)
            boot_musical = resample(musical_values, random_state=i+n_bootstrap)
            
            bootstrap_means_semantic.append(np.mean(boot_semantic))
            bootstrap_means_musical.append(np.mean(boot_musical))
            bootstrap_differences.append(np.mean(boot_musical) - np.mean(boot_semantic))
        
        # Análisis de distribuciones bootstrap
        bootstrap_analysis = {
            'n_bootstrap_samples': n_bootstrap,
            'difference_statistics': {
                'mean': float(np.mean(bootstrap_differences)),
                'std': float(np.std(bootstrap_differences)),
                'median': float(np.median(bootstrap_differences)),
                'percentile_2_5': float(np.percentile(bootstrap_differences, 2.5)),
                'percentile_97_5': float(np.percentile(bootstrap_differences, 97.5)),
                'proportion_positive': float(np.mean(np.array(bootstrap_differences) > 0))
            },
            'semantic_statistics': {
                'mean': float(np.mean(bootstrap_means_semantic)),
                'std': float(np.std(bootstrap_means_semantic))
            },
            'musical_statistics': {
                'mean': float(np.mean(bootstrap_means_musical)),
                'std': float(np.std(bootstrap_means_musical))
            },
            'bias_analysis': {
                'original_difference': float(np.mean(differences)),
                'bootstrap_bias': float(np.mean(bootstrap_differences) - np.mean(differences)),
                'bias_corrected_estimate': float(2 * np.mean(differences) - np.mean(bootstrap_differences))
            }
        }
        
        self.logger.info(f"Bootstrap completado: diferencia media {np.mean(bootstrap_differences):.4f}")
        self.logger.info(f"Proporción diferencias positivas: {bootstrap_analysis['difference_statistics']['proportion_positive']:.3f}")
        
        return bootstrap_analysis
    
    def _perform_normality_tests(self, hopkins_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar tests de normalidad para validar supuestos paramétricos.
        
        Args:
            hopkins_results: Resultados Hopkins
            
        Returns:
            Resultados de tests de normalidad
        """
        normality_analysis = {}
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            return normality_analysis
        
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        differences = np.array(musical_values) - np.array(semantic_values)
        
        # Tests para cada dataset
        datasets = {
            'semantic_values': semantic_values,
            'musical_values': musical_values,
            'differences': differences
        }
        
        for name, data in datasets.items():
            if len(data) < 3:  # Mínimo para tests
                continue
                
            dataset_tests = {}
            
            # Shapiro-Wilk test (mejor para n < 50)
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                dataset_tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal_strict': shapiro_p > self.alpha_levels['strict'],
                    'is_normal_standard': shapiro_p > self.alpha_levels['standard']
                }
            except Exception as e:
                self.logger.warning(f"Error en Shapiro-Wilk para {name}: {e}")
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
                dataset_tests['kolmogorov_smirnov'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal_strict': ks_p > self.alpha_levels['strict'],
                    'is_normal_standard': ks_p > self.alpha_levels['standard']
                }
            except Exception as e:
                self.logger.warning(f"Error en K-S para {name}: {e}")
            
            # Anderson-Darling test
            try:
                ad_result = stats.anderson(data, dist='norm')
                dataset_tests['anderson_darling'] = {
                    'statistic': float(ad_result.statistic),
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': ad_result.significance_level.tolist()
                }
            except Exception as e:
                self.logger.warning(f"Error en Anderson-Darling para {name}: {e}")
            
            normality_analysis[name] = dataset_tests
        
        return normality_analysis
    
    def _calculate_statistical_power(self, hopkins_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular poder estadístico del análisis.
        
        Args:
            hopkins_results: Resultados Hopkins
            
        Returns:
            Análisis de poder estadístico
        """
        power_analysis = {}
        
        comparative_metrics = hopkins_results.get('comparative_metrics', {})
        
        if not comparative_metrics:
            return power_analysis
        
        semantic_analysis = hopkins_results.get('semantic_analysis', {})
        musical_analysis = hopkins_results.get('musical_analysis', {})
        
        if not (semantic_analysis and musical_analysis):
            return power_analysis
        
        # Parámetros para cálculo de poder
        semantic_values = self._extract_iteration_values(semantic_analysis)
        musical_values = self._extract_iteration_values(musical_analysis)
        
        if not (semantic_values and musical_values):
            return power_analysis
        
        # Calcular efecto observado
        differences = np.array(musical_values) - np.array(semantic_values)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        n = len(differences)
        
        # Poder post-hoc aproximado usando distribución t no central
        try:
            from scipy.stats import nct
            
            # Parámetros
            alpha = 0.05
            df = n - 1
            ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
            
            # Valor crítico para test bilateral
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Poder como 1 - beta
            # Probabilidad de rechazar H0 cuando H1 es verdadera
            power_lower = nct.cdf(-t_critical, df, ncp)
            power_upper = 1 - nct.cdf(t_critical, df, ncp)
            power = power_lower + power_upper
            
            power_analysis['post_hoc_power'] = {
                'observed_effect_size': float(effect_size),
                'sample_size': n,
                'alpha': alpha,
                'power': float(power),
                'power_interpretation': self._interpret_statistical_power(power)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculando poder estadístico: {e}")
        
        # Cálculo de tamaño de muestra requerido para diferentes poderes
        desired_powers = [0.70, 0.80, 0.90, 0.95]
        required_sample_sizes = {}
        
        for desired_power in desired_powers:
            # Aproximación mediante iteración
            required_n = self._calculate_required_sample_size(effect_size, alpha, desired_power)
            required_sample_sizes[f'power_{int(desired_power*100)}'] = required_n
        
        power_analysis['sample_size_analysis'] = {
            'current_n': n,
            'required_sample_sizes': required_sample_sizes,
            'adequacy_assessment': self._assess_sample_size_adequacy(n, required_sample_sizes)
        }
        
        return power_analysis
    
    def _adjust_multiple_comparisons(self, hopkins_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplicar correcciones para comparaciones múltiples.
        
        Args:
            hopkins_results: Resultados Hopkins
            
        Returns:
            P-valores ajustados por comparaciones múltiples
        """
        # En este caso específico tenemos una comparación principal (semantic vs musical)
        # pero implementamos el framework para extensibilidad futura
        
        adjustment_analysis = {
            'method_applied': 'bonferroni_holm',
            'original_comparisons': 1,  # Solo comparación principal por ahora
            'adjustment_necessary': False,  # Con solo 1 comparación
            'adjusted_alpha_levels': {
                'strict': self.alpha_levels['strict'],
                'standard': self.alpha_levels['standard'],
                'exploratory': self.alpha_levels['exploratory']
            }
        }
        
        # Si en el futuro se agregan más comparaciones, aquí se implementaría
        # la corrección de Bonferroni-Holm u otros métodos
        
        return adjustment_analysis
    
    def _evaluate_conclusion_validity(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluar validez general de conclusiones basada en todos los tests.
        
        Args:
            validation_results: Todos los resultados de validación
            
        Returns:
            Evaluación de validez de conclusiones
        """
        validity_analysis = {
            'statistical_evidence_strength': 'insufficient',
            'convergent_validity': False,
            'assumption_violations': [],
            'recommendation': 'inconclusive',
            'confidence_in_conclusions': 'low'
        }
        
        # Analizar significancia estadística
        hopkins_sig = validation_results.get('hopkins_significance_analysis', {})
        
        significant_tests = 0
        total_tests = 0
        
        for test_name, test_results in hopkins_sig.items():
            if isinstance(test_results, dict) and 'p_value' in test_results:
                total_tests += 1
                if test_results.get('significant_standard', False):
                    significant_tests += 1
        
        # Evaluar tamaño de efecto
        effect_analysis = validation_results.get('effect_size_analysis', {})
        large_effect = False
        
        if effect_analysis.get('cohens_d_paired'):
            cohens_d = abs(effect_analysis['cohens_d_paired'])
            large_effect = cohens_d >= 0.8
        
        # Evaluar consistencia bootstrap
        bootstrap_analysis = validation_results.get('bootstrap_validation', {})
        bootstrap_consistent = False
        
        if bootstrap_analysis.get('difference_statistics'):
            proportion_positive = bootstrap_analysis['difference_statistics'].get('proportion_positive', 0)
            bootstrap_consistent = proportion_positive >= 0.95 or proportion_positive <= 0.05
        
        # Determinar fortaleza de evidencia
        if significant_tests >= total_tests * 0.67 and large_effect and bootstrap_consistent:
            validity_analysis['statistical_evidence_strength'] = 'strong'
            validity_analysis['confidence_in_conclusions'] = 'high'
            validity_analysis['recommendation'] = 'accept_conclusions'
        elif significant_tests >= total_tests * 0.5 and (large_effect or bootstrap_consistent):
            validity_analysis['statistical_evidence_strength'] = 'moderate'
            validity_analysis['confidence_in_conclusions'] = 'moderate'
            validity_analysis['recommendation'] = 'tentatively_accept_conclusions'
        else:
            validity_analysis['statistical_evidence_strength'] = 'weak'
            validity_analysis['confidence_in_conclusions'] = 'low'
            validity_analysis['recommendation'] = 'insufficient_evidence'
        
        # Evaluar validez convergente
        validity_analysis['convergent_validity'] = (
            significant_tests > 0 and 
            large_effect and 
            bootstrap_consistent
        )
        
        # Identificar violaciones de supuestos
        normality_results = validation_results.get('normality_tests', {})
        
        for dataset_name, tests in normality_results.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and not test_result.get('is_normal_standard', True):
                        validity_analysis['assumption_violations'].append(
                            f"Normalidad violada en {dataset_name} ({test_name})"
                        )
        
        return validity_analysis
    
    # Métodos auxiliares para cálculos estadísticos específicos
    def _extract_iteration_values(self, analysis_results: Dict[str, Any]) -> List[float]:
        """Extraer valores de iteraciones Hopkins."""
        all_iterations = analysis_results.get('all_iterations', [])
        
        if not all_iterations:
            return []
        
        values = []
        for iteration in all_iterations:
            if isinstance(iteration, dict) and 'hopkins_statistic' in iteration:
                values.append(iteration['hopkins_statistic'])
        
        return values
    
    def _calculate_cliff_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calcular Cliff's delta (medida de tamaño de efecto no paramétrica)."""
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Contar dominancias
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x2 > x1:
                    dominance += 1
                elif x2 < x1:
                    dominance -= 1
        
        return dominance / (n1 * n2)
    
    def _calculate_common_language_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calcular Common Language Effect Size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.5
        
        count = 0
        for x1 in group1:
            for x2 in group2:
                if x2 > x1:
                    count += 1
        
        return count / (n1 * n2)
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, confidence_level: float, 
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calcular intervalo de confianza bootstrap."""
        try:
            bootstrap_means = []
            
            for i in range(n_bootstrap):
                bootstrap_sample = resample(data, random_state=i)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            return (float(ci_lower), float(ci_upper))
            
        except Exception as e:
            self.logger.error(f"Error en bootstrap CI: {e}")
            return (0.0, 0.0)
    
    def _calculate_required_sample_size(self, effect_size: float, alpha: float, 
                                       desired_power: float) -> int:
        """Calcular tamaño de muestra requerido (aproximación)."""
        try:
            # Aproximación para test t bilateral
            from scipy.stats import norm
            
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(desired_power)
            
            # Fórmula aproximada
            n_required = ((z_alpha + z_beta) / effect_size) ** 2
            
            return max(3, int(np.ceil(n_required)))
            
        except Exception as e:
            self.logger.error(f"Error calculando tamaño muestra requerido: {e}")
            return 10  # Default conservativo
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpretar Cohen's d."""
        abs_d = abs(cohens_d)
        
        if abs_d >= 0.8:
            return "Large effect"
        elif abs_d >= 0.5:
            return "Medium effect"
        elif abs_d >= 0.2:
            return "Small effect"
        else:
            return "Negligible effect"
    
    def _interpret_cliff_delta(self, cliff_delta: float) -> str:
        """Interpretar Cliff's delta."""
        abs_delta = abs(cliff_delta)
        
        if abs_delta >= 0.474:
            return "Large effect"
        elif abs_delta >= 0.33:
            return "Medium effect"
        elif abs_delta >= 0.147:
            return "Small effect"
        else:
            return "Negligible effect"
    
    def _interpret_cles(self, cles: float) -> str:
        """Interpretar Common Language Effect Size."""
        if cles >= 0.71:
            return "Large effect (71%+ superiority)"
        elif cles >= 0.64:
            return "Medium effect (64%+ superiority)"
        elif cles >= 0.56:
            return "Small effect (56%+ superiority)"
        else:
            return "Negligible effect (~50% superiority)"
    
    def _interpret_statistical_power(self, power: float) -> str:
        """Interpretar poder estadístico."""
        if power >= 0.9:
            return "Excellent power (90%+)"
        elif power >= 0.8:
            return "Good power (80%+)"
        elif power >= 0.7:
            return "Adequate power (70%+)"
        else:
            return "Insufficient power (<70%)"
    
    def _assess_sample_size_adequacy(self, current_n: int, 
                                   required_sizes: Dict[str, int]) -> str:
        """Evaluar adecuación del tamaño de muestra actual."""
        required_80 = required_sizes.get('power_80', float('inf'))
        
        if current_n >= required_80:
            return "Adequate sample size for 80% power"
        elif current_n >= required_80* 0.7:
            return "Marginally adequate sample size"
        else:
            return "Insufficient sample size"