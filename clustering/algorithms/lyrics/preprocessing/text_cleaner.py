"""
Limpieza universal de letras musicales multilingües.
Optimizado para máxima calidad semántica + eficiencia BERT.

Características especializadas:
- Remoción metadatos estructurales musicales
- Manejo inteligente repeticiones
- Normalización Unicode robusta
- Optimización para modelos BERT
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from unidecode import unidecode

# Configurar logging
logger = logging.getLogger(__name__)

class MusicTextCleaner:
    """
    Limpieza especializada para letras musicales multilingües.
    
    Pipeline optimizado:
    1. Remoción metadatos estructurales
    2. Manejo inteligente repeticiones  
    3. Normalización Unicode
    4. Limpieza específica por idioma
    5. Optimización BERT
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa limpiador con configuración especializada.
        
        Args:
            config: Configuración personalizada. Si None, usa defaults.
        """
        self.config = config or self._get_default_config()
        self._compile_patterns()
        
        logger.debug("MusicTextCleaner inicializado con configuración personalizada")
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto optimizada para letras musicales."""
        return {
            "remove_structural_metadata": True,
            "handle_repetitions": "smart",
            "normalize_unicode": True,
            "preserve_emotional_words": True,
            "min_length_chars": 50,
            "max_length_chars": 5000,
            "target_tokens": 256,
            "remove_urls": True,
            "remove_numbers": False,  # Números pueden ser semánticamente relevantes
            "lowercase": True
        }
    
    def _compile_patterns(self):
        """Compila patrones regex para optimización performance."""
        
        # Patrones metadatos estructurales musicales
        self.structural_patterns = [
            # Etiquetas estructurales
            re.compile(r'\[(Verse|Chorus|Bridge|Intro|Outro|Hook|Pre-Chorus|Interlude).*?\]', re.IGNORECASE),
            re.compile(r'\[(V\d+|C\d+|B\d+|Intro|Outro)\]', re.IGNORECASE),
            re.compile(r'\[.*?(verse|chorus|bridge|intro|outro).*?\]', re.IGNORECASE),
            
            # Indicadores repetición
            re.compile(r'\(x\d+\)', re.IGNORECASE),
            re.compile(r'\(repeat.*?\)', re.IGNORECASE),
            re.compile(r'\[repeat.*?\]', re.IGNORECASE),
            
            # Metadata temporal
            re.compile(r'\d+:\d+', re.IGNORECASE),  # timestamps
            re.compile(r'\[[\d:]+\]', re.IGNORECASE),  # [0:45]
            
            # Anotaciones productores/artistas
            re.compile(r'\(feat\..*?\)', re.IGNORECASE),
            re.compile(r'\(ft\..*?\)', re.IGNORECASE),
            re.compile(r'\(produced by.*?\)', re.IGNORECASE)
        ]
        
        # Patrones interjecciones musicales comunes
        self.interjection_pattern = re.compile(
            r'\b(?:yeah|oh|ah|eh|uh|mm|hmm|hey|yo|whoa|ooh|aah|'
            r'la|da|na|ba|sha|doo|woo|tra|ho|ha|ay|ey|oy)\b',
            re.IGNORECASE
        )
        
        # Patrones repetición líneas
        self.repetition_patterns = [
            # Repetición exacta líneas
            re.compile(r'^(.+)(\n\1){2,}', re.MULTILINE),  # 3+ repeticiones idénticas
            re.compile(r'^(.{10,})(\n\1)+', re.MULTILINE),  # 2+ repeticiones líneas largas
        ]
        
        # Patrones URLs y elementos web
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        
        # Patrones espaciado y puntuación
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s\'-]')
        
        logger.debug("Patrones regex compilados para optimización performance")
    
    def clean_universal(self, text: str, language: Optional[str] = None) -> str:
        """
        Pipeline limpieza universal multi-etapa.
        
        Args:
            text: Texto lyrics a limpiar
            language: Código idioma (en, es, de, pt). Si None, detecta automáticamente
            
        Returns:
            Texto limpio optimizado para BERT
        """
        if not text or not isinstance(text, str):
            logger.warning("Texto vacío o inválido recibido")
            return ""
        
        original_length = len(text)
        logger.debug(f"Iniciando limpieza texto: {original_length} caracteres")
        
        # 1. Limpieza estructural musical
        text = self._remove_musical_metadata(text)
        
        # 2. Manejo inteligente repeticiones
        text = self._handle_repetitions_smart(text)
        
        # 3. Normalización Unicode
        if self.config["normalize_unicode"]:
            text = self._normalize_unicode(text)
        
        # 4. Limpieza general
        text = self._clean_general(text)
        
        # 5. Limpieza específica idioma
        if language:
            text = self._clean_language_specific(text, language)
        
        # 6. Optimización BERT
        text = self._optimize_for_bert(text)
        
        # 7. Validación final
        text = self._validate_and_finalize(text)
        
        final_length = len(text)
        reduction_pct = ((original_length - final_length) / original_length) * 100
        
        logger.debug(f"Limpieza completada: {final_length} caracteres "
                    f"({reduction_pct:.1f}% reducción)")
        
        return text
    
    def _remove_musical_metadata(self, text: str) -> str:
        """Remueve metadatos estructurales específicos música."""
        if not self.config["remove_structural_metadata"]:
            return text
        
        # Aplicar todos los patrones estructurales
        for pattern in self.structural_patterns:
            text = pattern.sub(' ', text)
        
        logger.debug("Metadatos estructurales musicales removidos")
        return text
    
    def _handle_repetitions_smart(self, text: str) -> str:
        """
        Manejo inteligente repeticiones preservando semántica.
        
        Estrategia:
        - Chorus/Estribillo: preservar 1 ocurrencia completa
        - Línea repetida: preservar 2 primeras ocurrencias  
        - Palabra énfasis: preservar hasta 3 repeticiones
        """
        if self.config["handle_repetitions"] != "smart":
            return text
        
        lines = text.split('\n')
        processed_lines = []
        line_counts = {}
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Contar ocurrencias de cada línea
            if line_clean in line_counts:
                line_counts[line_clean] += 1
            else:
                line_counts[line_clean] = 1
            
            # Estrategia preservación basada en longitud línea
            if len(line_clean) > 50:  # Línea larga (probablemente verso completo)
                if line_counts[line_clean] <= 1:  # Solo primera ocurrencia
                    processed_lines.append(line_clean)
            elif len(line_clean) > 20:  # Línea mediana  
                if line_counts[line_clean] <= 2:  # Hasta 2 ocurrencias
                    processed_lines.append(line_clean)
            else:  # Línea corta (posible interjección)
                if line_counts[line_clean] <= 3:  # Hasta 3 ocurrencias
                    processed_lines.append(line_clean)
        
        result = '\n'.join(processed_lines)
        logger.debug(f"Repeticiones manejadas: {len(lines)} -> {len(processed_lines)} líneas")
        
        return result
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalización Unicode robusta manteniendo significado."""
        import unicodedata
        
        # Normalización NFD -> NFC para consistencia
        text = unicodedata.normalize('NFC', text)
        
        # Preservar algunos caracteres especiales importantes para significado
        preserve_chars = {'ñ', 'ç', 'ü', 'ö', 'ä', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ì', 'ò', 'ù'}
        
        # Solo usar unidecode para caracteres realmente problemáticos
        result = ""
        for char in text:
            if char.isascii() or char in preserve_chars:
                result += char
            else:
                # Usar unidecode solo para caracteres exóticos
                result += unidecode(char)
        
        logger.debug("Normalización Unicode aplicada preservando caracteres significativos")
        return result
    
    def _clean_general(self, text: str) -> str:
        """Limpieza general aplicable a todos idiomas."""
        
        # Remover URLs si configurado
        if self.config["remove_urls"]:
            text = self.url_pattern.sub(' ', text)
        
        # Remover números si configurado
        if self.config["remove_numbers"]:
            text = re.sub(r'\b\d+\b', ' ', text)
        
        # Normalizar espacios múltiples
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remover líneas vacías múltiples
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Lowercase si configurado
        if self.config["lowercase"]:
            text = text.lower()
        
        logger.debug("Limpieza general aplicada")
        return text.strip()
    
    def _clean_language_specific(self, text: str, language: str) -> str:
        """
        Limpieza específica por idioma.
        
        Args:
            text: Texto a limpiar
            language: Código idioma (en, es, de, pt)
            
        Returns:
            Texto limpio con reglas específicas idioma
        """
        
        if language == "en":
            # Expansión contracciones inglés
            contractions = {
                "can't": "cannot", "won't": "will not", "n't": " not",
                "I'm": "I am", "you're": "you are", "we're": "we are",
                "they're": "they are", "I've": "I have", "you've": "you have",
                "we've": "we have", "they've": "they have", "I'll": "I will",
                "you'll": "you will", "we'll": "we will", "they'll": "they will",
                "I'd": "I would", "you'd": "you would", "we'd": "we would",
                "they'd": "they would", "'s": " is", "'re": " are"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
                
        elif language == "es":
            # Normalización contracciones español
            text = text.replace("del ", "de el ")
            text = text.replace("al ", "a el ")
            
        elif language == "de":
            # Normalización umlauts alemán si necesario (ya preservados en Unicode)
            pass
            
        elif language == "pt":
            # Normalización contracciones portugués
            text = text.replace("do ", "de o ")
            text = text.replace("da ", "de a ")
            text = text.replace("num ", "em um ")
        
        logger.debug(f"Limpieza específica idioma {language} aplicada")
        return text
    
    def _optimize_for_bert(self, text: str) -> str:
        """Optimización específica para modelos BERT."""
        
        # Validar longitud para BERT (target 256 tokens ≈ 1000-1200 chars)
        max_chars = self.config["max_length_chars"]
        
        if len(text) > max_chars:
            # Truncation inteligente: preservar inicio + final
            mid_point = max_chars // 2
            start_part = text[:mid_point-100]
            end_part = text[-(mid_point-100):]
            text = start_part + " ... " + end_part
            
            logger.debug(f"Texto truncado inteligentemente a {len(text)} caracteres")
        
        # Asegurar que no queden espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _validate_and_finalize(self, text: str) -> str:
        """Validación final y control calidad."""
        
        # Verificar longitud mínima
        if len(text) < self.config["min_length_chars"]:
            logger.warning(f"Texto muy corto después limpieza: {len(text)} caracteres")
            return ""
        
        # Verificar que no sea solo espacios/puntuación
        if not re.search(r'[a-zA-Z]', text):
            logger.warning("No quedan caracteres alfabéticos después limpieza")
            return ""
        
        # Calcular diversidad léxica básica
        words = text.split()
        if len(words) > 0:
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            
            if diversity < 0.3:  # TTR muy bajo
                logger.warning(f"Diversidad léxica baja: {diversity:.2f}")
        
        logger.debug("Validación final completada exitosamente")
        return text
    
    def clean_batch(self, texts: List[str], languages: Optional[List[str]] = None) -> List[str]:
        """
        Limpieza batch optimizada para múltiples textos.
        
        Args:
            texts: Lista textos a limpiar
            languages: Lista códigos idioma correspondientes
            
        Returns:
            Lista textos limpios
        """
        if not texts:
            return []
        
        if languages and len(languages) != len(texts):
            logger.warning("Número idiomas no coincide con número textos")
            languages = None
        
        results = []
        failed_count = 0
        
        for i, text in enumerate(texts):
            try:
                language = languages[i] if languages else None
                cleaned = self.clean_universal(text, language)
                results.append(cleaned)
                
                if not cleaned:  # Texto quedó vacío
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error limpiando texto {i}: {e}")
                results.append("")
                failed_count += 1
        
        success_rate = ((len(texts) - failed_count) / len(texts)) * 100
        logger.info(f"Batch limpieza completada: {success_rate:.1f}% éxito "
                   f"({len(texts) - failed_count}/{len(texts)})")
        
        return results
    
    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict:
        """
        Obtiene estadísticas detalladas del proceso limpieza.
        
        Args:
            original_text: Texto original
            cleaned_text: Texto después limpieza
            
        Returns:
            Dict con estadísticas detalladas
        """
        stats = {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "reduction_percentage": 0,
            "original_words": 0,
            "cleaned_words": 0,
            "word_reduction_percentage": 0,
            "original_lines": 0,
            "cleaned_lines": 0,
            "diversity_original": 0,
            "diversity_cleaned": 0
        }
        
        if original_text:
            original_words = original_text.split()
            stats["original_words"] = len(original_words)
            stats["original_lines"] = len(original_text.split('\n'))
            
            if original_words:
                stats["diversity_original"] = len(set(original_words)) / len(original_words)
        
        if cleaned_text:
            cleaned_words = cleaned_text.split()
            stats["cleaned_words"] = len(cleaned_words)
            stats["cleaned_lines"] = len(cleaned_text.split('\n'))
            
            if cleaned_words:
                stats["diversity_cleaned"] = len(set(cleaned_words)) / len(cleaned_words)
        
        if stats["original_length"] > 0:
            stats["reduction_percentage"] = (
                (stats["original_length"] - stats["cleaned_length"]) / 
                stats["original_length"]
            ) * 100
        
        if stats["original_words"] > 0:
            stats["word_reduction_percentage"] = (
                (stats["original_words"] - stats["cleaned_words"]) / 
                stats["original_words"]
            ) * 100
        
        return stats