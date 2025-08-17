"""
Extractor de características de texto para optimización BERT.
Análisis calidad texto y preparación para vectorización.

Funcionalidades:
- Análisis longitud y complejidad texto
- Cálculo diversidad léxica (TTR)
- Detección idioma y validación
- Preparación batches BERT optimizados
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import statistics

logger = logging.getLogger(__name__)

class TextFeatureExtractor:
    """
    Extractor características texto para optimización clustering semántico.
    
    Analiza:
    - Estadísticas básicas texto (longitud, palabras, líneas)
    - Diversidad léxica (TTR, vocabulary richness)
    - Complejidad sintáctica y semántica
    - Calidad texto para BERT processing
    """
    
    def __init__(self):
        """Inicializa extractor con configuraciones por defecto."""
        self._compile_patterns()
        logger.debug("TextFeatureExtractor inicializado")
    
    def _compile_patterns(self):
        """Compila patrones regex para análisis eficiente."""
        
        # Patrón palabras (alphanumeric + apostrophes)
        self.word_pattern = re.compile(r"\b\w+(?:'\w+)?\b")
        
        # Patrón oraciones (basado en puntuación)
        self.sentence_pattern = re.compile(r'[.!?]+')
        
        # Patrón repeticiones líneas
        self.line_repetition_pattern = re.compile(r'^(.+)$.*^\1$', re.MULTILINE)
        
        # Patrón caracteres especiales
        self.special_chars_pattern = re.compile(r'[^\w\s\'-]')
        
        logger.debug("Patrones regex compilados para extracción características")
    
    def extract_basic_features(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Extrae características básicas del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con características básicas
        """
        if not text:
            return self._empty_features_dict()
        
        # Estadísticas básicas
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        line_count = len(text.split('\n'))
        non_empty_lines = len([line for line in text.split('\n') if line.strip()])
        
        # Análisis palabras
        words = self.word_pattern.findall(text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        
        # Análisis oraciones
        sentences = self.sentence_pattern.split(text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Cálculos derivados
        avg_word_length = statistics.mean(len(word) for word in words) if words else 0
        avg_words_per_line = word_count / non_empty_lines if non_empty_lines > 0 else 0
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else word_count
        
        features = {
            # Conteos básicos
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "unique_word_count": unique_words,
            "line_count": line_count,
            "non_empty_line_count": non_empty_lines,
            "sentence_count": sentence_count,
            
            # Métricas derivadas
            "avg_word_length": round(avg_word_length, 2),
            "avg_words_per_line": round(avg_words_per_line, 2),
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "chars_per_word": round(char_count / word_count, 2) if word_count > 0 else 0,
            
            # Ratios útiles
            "space_ratio": round((char_count - char_count_no_spaces) / char_count, 3) if char_count > 0 else 0,
            "empty_line_ratio": round((line_count - non_empty_lines) / line_count, 3) if line_count > 0 else 0
        }
        
        logger.debug(f"Características básicas extraídas: {word_count} palabras, {unique_words} únicas")
        return features
    
    def calculate_lexical_diversity(self, text: str) -> Dict[str, float]:
        """
        Calcula métricas diversidad léxica.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con métricas diversidad
        """
        if not text:
            return {"ttr": 0.0, "rttr": 0.0, "cttr": 0.0, "mtld": 0.0}
        
        words = self.word_pattern.findall(text.lower())
        if not words:
            return {"ttr": 0.0, "rttr": 0.0, "cttr": 0.0, "mtld": 0.0}
        
        word_count = len(words)
        unique_count = len(set(words))
        
        # TTR (Type-Token Ratio) - básico
        ttr = unique_count / word_count
        
        # RTTR (Root TTR) - más estable con textos largos
        rttr = unique_count / (word_count ** 0.5)
        
        # CTTR (Corrected TTR) - normalizado
        cttr = unique_count / (2 * word_count) ** 0.5
        
        # MTLD (Measure of Textual Lexical Diversity) - aproximación simplificada
        mtld = self._calculate_mtld_approx(words)
        
        diversity_metrics = {
            "ttr": round(ttr, 3),
            "rttr": round(rttr, 3), 
            "cttr": round(cttr, 3),
            "mtld": round(mtld, 3)
        }
        
        logger.debug(f"Diversidad léxica: TTR={ttr:.3f}, RTTR={rttr:.3f}")
        return diversity_metrics
    
    def _calculate_mtld_approx(self, words: List[str]) -> float:
        """
        Cálculo aproximado MTLD (Measure of Textual Lexical Diversity).
        
        Args:
            words: Lista palabras tokenizadas
            
        Returns:
            MTLD score aproximado
        """
        if len(words) < 50:  # MTLD no confiable para textos muy cortos
            return 0.0
        
        # Dividir texto en segmentos y calcular TTR promedio
        segment_size = 50
        segments = [words[i:i+segment_size] for i in range(0, len(words), segment_size)]
        
        ttr_scores = []
        for segment in segments:
            if len(segment) >= 10:  # Mínimo para cálculo confiable
                ttr = len(set(segment)) / len(segment)
                ttr_scores.append(ttr)
        
        if not ttr_scores:
            return 0.0
        
        # MTLD aproximado como inverso de TTR promedio
        avg_ttr = statistics.mean(ttr_scores)
        mtld_approx = 1 / avg_ttr if avg_ttr > 0 else 0.0
        
        return mtld_approx
    
    def analyze_repetition_patterns(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Analiza patrones repetición en el texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con métricas repetición
        """
        if not text:
            return {"line_repetitions": 0, "word_repetitions": 0, "repetition_ratio": 0.0}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = self.word_pattern.findall(text.lower())
        
        # Contar repeticiones líneas
        line_counts = Counter(lines)
        repeated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        # Contar repeticiones palabras
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        # Ratio repetición general
        total_tokens = len(lines) + len(words)
        total_repetitions = repeated_lines + repeated_words
        repetition_ratio = total_repetitions / total_tokens if total_tokens > 0 else 0.0
        
        repetition_metrics = {
            "line_repetitions": repeated_lines,
            "word_repetitions": repeated_words,
            "repetition_ratio": round(repetition_ratio, 3),
            "unique_lines_ratio": round(len(set(lines)) / len(lines), 3) if lines else 0.0,
            "most_repeated_word_count": max(word_counts.values()) if word_counts else 0
        }
        
        logger.debug(f"Análisis repetición: {repeated_lines} líneas, {repeated_words} palabras")
        return repetition_metrics
    
    def assess_text_quality(self, text: str, language: str = None) -> Dict[str, Union[bool, float, str]]:
        """
        Evalúa calidad general del texto para processing BERT.
        
        Args:
            text: Texto a evaluar
            language: Idioma esperado (opcional)
            
        Returns:
            Dict con assessment calidad
        """
        if not text:
            return {"quality_score": 0.0, "is_suitable": False, "issues": ["empty_text"]}
        
        issues = []
        quality_factors = []
        
        # 1. Análisis longitud
        char_count = len(text)
        word_count = len(self.word_pattern.findall(text))
        
        if char_count < 50:
            issues.append("too_short")
            quality_factors.append(0.0)
        elif char_count > 5000:
            issues.append("too_long")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)
        
        # 2. Análisis diversidad léxica
        diversity = self.calculate_lexical_diversity(text)
        ttr = diversity["ttr"]
        
        if ttr < 0.3:
            issues.append("low_diversity")
            quality_factors.append(ttr / 0.3)  # Penalización proporcional
        else:
            quality_factors.append(1.0)
        
        # 3. Análisis repetición
        repetition = self.analyze_repetition_patterns(text)
        repetition_ratio = repetition["repetition_ratio"]
        
        if repetition_ratio > 0.7:
            issues.append("excessive_repetition")
            quality_factors.append(1.0 - repetition_ratio)
        else:
            quality_factors.append(1.0)
        
        # 4. Análisis contenido alfabético
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text) if text else 0
        
        if alpha_ratio < 0.5:
            issues.append("low_alphabetic_content")
            quality_factors.append(alpha_ratio / 0.5)
        else:
            quality_factors.append(1.0)
        
        # 5. Detección idioma si especificado
        if language:
            detected_lang = self._detect_language_simple(text)
            if detected_lang != language and detected_lang != "unknown":
                issues.append(f"language_mismatch_expected_{language}_detected_{detected_lang}")
                quality_factors.append(0.8)  # Penalización moderada
            else:
                quality_factors.append(1.0)
        
        # Calcular score final
        quality_score = statistics.mean(quality_factors) if quality_factors else 0.0
        is_suitable = quality_score >= 0.6 and len(issues) <= 2
        
        assessment = {
            "quality_score": round(quality_score, 3),
            "is_suitable": is_suitable,
            "issues": issues,
            "word_count": word_count,
            "char_count": char_count,
            "diversity_ttr": round(ttr, 3),
            "repetition_ratio": round(repetition_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3)
        }
        
        logger.debug(f"Calidad texto: score={quality_score:.3f}, suitable={is_suitable}")
        return assessment
    
    def _detect_language_simple(self, text: str) -> str:
        """
        Detección simple idioma basada en características básicas.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Código idioma detectado o "unknown"
        """
        text_lower = text.lower()
        
        # Indicadores simples por idioma
        language_indicators = {
            "en": ["the", "and", "you", "that", "was", "for", "are", "with", "his", "they"],
            "es": ["que", "del", "las", "una", "con", "por", "sus", "las", "como", "pero"],
            "de": ["der", "die", "und", "den", "das", "ich", "ist", "mit", "sie", "auf"],
            "pt": ["que", "uma", "com", "não", "das", "dos", "por", "mais", "mas", "seu"]
        }
        
        scores = {}
        words = self.word_pattern.findall(text_lower)
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            scores[lang] = score / len(words) if words else 0
        
        if not scores or max(scores.values()) < 0.02:  # Threshold muy bajo
            return "unknown"
        
        return max(scores, key=scores.get)
    
    def prepare_for_bert(self, text: str, max_tokens: int = 256) -> Dict[str, Union[str, bool, int]]:
        """
        Prepara texto para processing BERT óptimo.
        
        Args:
            text: Texto a preparar
            max_tokens: Máximo tokens BERT (aprox 3.5 chars = 1 token)
            
        Returns:
            Dict con texto preparado y metadata
        """
        if not text:
            return {"prepared_text": "", "is_truncated": False, "estimated_tokens": 0}
        
        # Estimación tokens (aproximada: 3.5 chars promedio = 1 token)
        estimated_tokens = len(text) / 3.5
        max_chars = int(max_tokens * 3.5)
        
        prepared_text = text
        is_truncated = False
        
        if estimated_tokens > max_tokens:
            # Truncation inteligente: preservar inicio + final
            if len(text) > max_chars:
                mid_point = max_chars // 2
                start_part = text[:mid_point-20]
                end_part = text[-(mid_point-20):]
                prepared_text = start_part + " ... " + end_part
                is_truncated = True
        
        # Limpieza final espacios
        prepared_text = re.sub(r'\s+', ' ', prepared_text).strip()
        
        final_estimated_tokens = len(prepared_text) / 3.5
        
        result = {
            "prepared_text": prepared_text,
            "is_truncated": is_truncated,
            "estimated_tokens": int(final_estimated_tokens),
            "original_estimated_tokens": int(estimated_tokens),
            "reduction_ratio": round((estimated_tokens - final_estimated_tokens) / estimated_tokens, 3) if estimated_tokens > 0 else 0.0
        }
        
        logger.debug(f"Texto preparado BERT: {int(estimated_tokens)} -> {int(final_estimated_tokens)} tokens")
        return result
    
    def extract_all_features(self, text: str, language: str = None) -> Dict:
        """
        Extrae todas las características disponibles del texto.
        
        Args:
            text: Texto a analizar
            language: Idioma esperado (opcional)
            
        Returns:
            Dict completo con todas las características
        """
        all_features = {}
        
        # Características básicas
        all_features.update(self.extract_basic_features(text))
        
        # Diversidad léxica
        diversity = self.calculate_lexical_diversity(text)
        all_features.update({f"diversity_{k}": v for k, v in diversity.items()})
        
        # Patrones repetición
        repetition = self.analyze_repetition_patterns(text)
        all_features.update({f"repetition_{k}": v for k, v in repetition.items()})
        
        # Assessment calidad
        quality = self.assess_text_quality(text, language)
        all_features.update({f"quality_{k}": v for k, v in quality.items()})
        
        # Preparación BERT
        bert_prep = self.prepare_for_bert(text)
        all_features.update({f"bert_{k}": v for k, v in bert_prep.items()})
        
        logger.debug("Extracción completa características finalizada")
        return all_features
    
    def _empty_features_dict(self) -> Dict[str, Union[int, float]]:
        """Retorna dict vacío con estructura características básicas."""
        return {
            "char_count": 0, "char_count_no_spaces": 0, "word_count": 0,
            "unique_word_count": 0, "line_count": 0, "non_empty_line_count": 0,
            "sentence_count": 0, "avg_word_length": 0.0, "avg_words_per_line": 0.0,
            "avg_words_per_sentence": 0.0, "chars_per_word": 0.0,
            "space_ratio": 0.0, "empty_line_ratio": 0.0
        }

def extract_text_features(text: str, language: str = None) -> Dict:
    """
    Función helper para extracción rápida características.
    
    Args:
        text: Texto a analizar
        language: Idioma esperado (opcional)
        
    Returns:
        Dict con características completas del texto
    """
    extractor = TextFeatureExtractor()
    return extractor.extract_all_features(text, language)