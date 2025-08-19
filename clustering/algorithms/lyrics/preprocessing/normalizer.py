"""
Normalización Unicode y específica por idioma.
Manejo robusto caracteres especiales multilingües.

Especializado para:
- Normalización Unicode preservando significado
- Expansión contracciones por idioma
- Manejo dialéctico regional
- Consistencia cross-lingual
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple
from unidecode import unidecode

logger = logging.getLogger(__name__)

class MultilingualNormalizer:
    """
    Normalizador especializado para letras musicales multilingües.
    
    Funcionalidades:
    - Normalización Unicode inteligente
    - Expansión contracciones por idioma
    - Manejo dialectos y regionalismos
    - Preservación significado semántico
    """
    
    def __init__(self):
        """Inicializa normalizador con configuraciones idioma."""
        self._load_language_configs()
        self._compile_patterns()
        
        logger.debug("MultilingualNormalizer inicializado")
    
    def _load_language_configs(self):
        """Carga configuraciones específicas por idioma."""
        
        # Contracciones inglés (más completas)
        self.english_contractions = {
            # Negaciones
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "isn't": "is not", "mustn't": "must not",
            "needn't": "need not", "shouldn't": "should not", "wasn't": "was not",
            "weren't": "were not", "won't": "will not", "wouldn't": "would not",
            
            # Pronombres + verbos
            "I'm": "I am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "I've": "I have", "you've": "you have", "we've": "we have",
            "they've": "they have", "I'll": "I will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "I'd": "I would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would",
            
            # Slang musical común
            "gonna": "going to", "wanna": "want to", "gotta": "got to",
            "hafta": "have to", "shoulda": "should have", "coulda": "could have",
            "woulda": "would have", "lemme": "let me", "gimme": "give me",
            "dunno": "do not know", "kinda": "kind of", "sorta": "sort of",
            
            # Informales adicionales
            "ya": "you", "em": "them", "cause": "because", "'til": "until",
            "bout": "about", "round": "around", "fore": "before"
        }
        
        # Contracciones español
        self.spanish_contractions = {
            "del": "de el", "al": "a el", "pa": "para", "to": "todo",
            "na": "nada", "tú": "tu",  # Normalización acentos opcionales
            "má": "más", "sí": "si"
        }
        
        # Regionalismos español (normalización)
        self.spanish_regionalisms = {
            "vos": "tú", "che": "amigo", "pibe": "chico", "chaval": "chico",
            "tío": "amigo", "colega": "amigo", "pana": "amigo", "bro": "hermano",
            "wey": "amigo", "güey": "amigo", "maje": "amigo"
        }
        
        # Contracciones alemán
        self.german_contractions = {
            "ich hab": "ich habe", "du hast": "du hast", "er hat": "er hat",
            "wir haben": "wir haben", "ihr habt": "ihr habt", "sie haben": "sie haben",
            "bin": "bin", "bist": "bist", "ist": "ist", "sind": "sind", "seid": "seid"
        }
        
        # Contracciones portugués
        self.portuguese_contractions = {
            "do": "de o", "da": "de a", "dos": "de os", "das": "de as",
            "no": "em o", "na": "em a", "nos": "em os", "nas": "em as",
            "pelo": "por o", "pela": "por a", "pelos": "por os", "pelas": "por as",
            "dum": "de um", "duma": "de uma", "duns": "de uns", "dumas": "de umas",
            "num": "em um", "numa": "em uma", "nuns": "em uns", "numas": "em umas",
            "pro": "para o", "pra": "para a", "pros": "para os", "pras": "para as"
        }
        
        # Caracteres especiales preservar por idioma
        self.preserve_chars = {
            "en": set(),  # Inglés no necesita caracteres especiales
            "es": {"ñ", "á", "é", "í", "ó", "ú", "ü", "ç"},
            "de": {"ä", "ö", "ü", "ß"},
            "pt": {"ã", "á", "à", "â", "ç", "é", "ê", "í", "ó", "ô", "õ", "ú", "ü"}
        }
        
        logger.debug("Configuraciones idiomas cargadas")
    
    def _compile_patterns(self):
        """Compila patrones regex para optimización."""
        
        # Patrón dialectos/variaciones ortográficas
        self.dialect_patterns = {
            "en": [
                (re.compile(r'\bcolou?r\b', re.IGNORECASE), 'color'),  # colour -> color
                (re.compile(r'\bcentre\b', re.IGNORECASE), 'center'),  # centre -> center
                (re.compile(r'\brealise\b', re.IGNORECASE), 'realize'), # realise -> realize
            ],
            "es": [
                (re.compile(r'\bokay\b', re.IGNORECASE), 'está bien'),  # anglicismos
                (re.compile(r'\bok\b', re.IGNORECASE), 'bien'),
            ]
        }
        
        # Patrón espacios múltiples
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Patrón puntuación múltiple
        self.punct_pattern = re.compile(r'([.!?]){2,}')
        
        logger.debug("Patrones regex compilados")
    
    def normalize_by_language(self, text: str, language: str) -> str:
        """
        Normalización específica por idioma.
        
        Args:
            text: Texto a normalizar
            language: Código idioma (en, es, de, pt)
            
        Returns:
            Texto normalizado según reglas idioma
        """
        if not text:
            return ""
        
        logger.debug(f"Normalizando texto en idioma: {language}")
        
        # 1. Normalización Unicode preservando caracteres importantes
        text = self._normalize_unicode_smart(text, language)
        
        # 2. Expansión contracciones específicas idioma
        text = self._expand_contractions(text, language)
        
        # 3. Normalización dialéctica
        text = self._normalize_dialects(text, language)
        
        # 4. Limpieza final
        text = self._final_cleanup(text)
        
        logger.debug(f"Normalización {language} completada")
        return text
    
    def _normalize_unicode_smart(self, text: str, language: str) -> str:
        """
        Normalización Unicode inteligente preservando significado.
        
        Args:
            text: Texto a normalizar
            language: Idioma para preservar caracteres específicos
            
        Returns:
            Texto con Unicode normalizado
        """
        # Normalización NFD -> NFC para consistencia
        text = unicodedata.normalize('NFC', text)
        
        # Obtener caracteres a preservar para este idioma
        preserve_chars = self.preserve_chars.get(language, set())
        
        result = ""
        for char in text:
            if char.isascii():
                # Caracteres ASCII siempre se preservan
                result += char
            elif char in preserve_chars:
                # Caracteres importantes del idioma se preservan
                result += char
            elif unicodedata.category(char) in ['Mn', 'Mc']:
                # Marcas diacríticas - intentar preservar si son significativas
                result += char
            else:
                # Otros caracteres - usar unidecode para ASCII equivalente
                ascii_equiv = unidecode(char)
                result += ascii_equiv if ascii_equiv else char
        
        logger.debug(f"Unicode normalizado preservando {len(preserve_chars)} caracteres {language}")
        return result
    
    def _expand_contractions(self, text: str, language: str) -> str:
        """
        Expande contracciones específicas del idioma.
        
        Args:
            text: Texto con contracciones
            language: Idioma para aplicar reglas específicas
            
        Returns:
            Texto con contracciones expandidas
        """
        contractions_map = {
            "en": self.english_contractions,
            "es": self.spanish_contractions,
            "de": self.german_contractions,
            "pt": self.portuguese_contractions
        }
        
        contractions = contractions_map.get(language, {})
        if not contractions:
            return text
        
        # Aplicar contracciones en orden de longitud (más largas primero)
        sorted_contractions = sorted(contractions.items(), key=lambda x: len(x[0]), reverse=True)
        
        for contraction, expansion in sorted_contractions:
            # Usar word boundaries para evitar reemplazos parciales
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        logger.debug(f"Contracciones {language} expandidas: {len(contractions)} reglas")
        return text
    
    def _normalize_dialects(self, text: str, language: str) -> str:
        """
        Normaliza dialectos y regionalismos.
        
        Args:
            text: Texto con variaciones dialécticas
            language: Idioma para aplicar normalizaciones
            
        Returns:
            Texto con dialectos normalizados
        """
        # Aplicar regionalismos específicos
        if language == "es":
            for regional, standard in self.spanish_regionalisms.items():
                pattern = r'\b' + re.escape(regional) + r'\b'
                text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        
        # Aplicar patrones dialécticos compilados
        dialect_patterns = self.dialect_patterns.get(language, [])
        for pattern, replacement in dialect_patterns:
            text = pattern.sub(replacement, text)
        
        logger.debug(f"Dialectos {language} normalizados")
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Limpieza final normalización."""
        
        # Normalizar espacios múltiples
        text = self.whitespace_pattern.sub(' ', text)
        
        # Normalizar puntuación múltiple
        text = self.punct_pattern.sub(r'\1', text)
        
        # Remover espacios al inicio/final líneas
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    def normalize_batch(self, texts: List[str], languages: List[str]) -> List[str]:
        """
        Normalización batch optimizada.
        
        Args:
            texts: Lista textos a normalizar
            languages: Lista códigos idioma correspondientes
            
        Returns:
            Lista textos normalizados
        """
        if len(texts) != len(languages):
            raise ValueError("Número textos debe coincidir con número idiomas")
        
        results = []
        language_counts = {}
        
        for text, language in zip(texts, languages):
            try:
                normalized = self.normalize_by_language(text, language)
                results.append(normalized)
                
                # Contar procesamiento por idioma
                language_counts[language] = language_counts.get(language, 0) + 1
                
            except Exception as e:
                logger.error(f"Error normalizando texto en {language}: {e}")
                results.append(text)  # Fallback al original
        
        logger.info(f"Batch normalización completada: {language_counts}")
        return results
    
    def normalize_auto_detect(self, text: str) -> str:
        """
        Detecta idioma automáticamente y normaliza (solo retorna texto).
        
        Args:
            text: Texto a procesar
            
        Returns:
            Texto normalizado
        """
        normalized_text, _ = self.detect_and_normalize(text)
        return normalized_text
    
    def detect_and_normalize(self, text: str) -> Tuple[str, str]:
        """
        Detecta idioma automáticamente y normaliza.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Tuple (texto_normalizado, idioma_detectado)
        """
        try:
            from langdetect import detect
            detected_language = detect(text)
            
            # Mapear códigos langdetect a nuestros códigos
            language_mapping = {
                "en": "en", "es": "es", "de": "de", "pt": "pt",
                "ca": "es",  # Catalán -> Español
                "gl": "es",  # Gallego -> Español  
                "fr": "es"   # Francés -> Español (similar)
            }
            
            normalized_lang = language_mapping.get(detected_language, "en")
            normalized_text = self.normalize_by_language(text, normalized_lang)
            
            logger.debug(f"Idioma detectado: {detected_language} -> {normalized_lang}")
            return normalized_text, normalized_lang
            
        except Exception as e:
            logger.warning(f"Error detección idioma: {e}. Usando inglés por defecto")
            normalized_text = self.normalize_by_language(text, "en")
            return normalized_text, "en"
    
    def get_supported_languages(self) -> List[str]:
        """Retorna lista idiomas soportados."""
        return ["en", "es", "de", "pt"]
    
    def get_language_stats(self) -> Dict[str, Dict]:
        """Retorna estadísticas configuraciones por idioma."""
        return {
            "en": {
                "contractions": len(self.english_contractions),
                "preserved_chars": len(self.preserve_chars["en"]),
                "dialect_patterns": len(self.dialect_patterns.get("en", []))
            },
            "es": {
                "contractions": len(self.spanish_contractions),
                "regionalisms": len(self.spanish_regionalisms),
                "preserved_chars": len(self.preserve_chars["es"]),
                "dialect_patterns": len(self.dialect_patterns.get("es", []))
            },
            "de": {
                "contractions": len(self.german_contractions),
                "preserved_chars": len(self.preserve_chars["de"]),
                "dialect_patterns": len(self.dialect_patterns.get("de", []))
            },
            "pt": {
                "contractions": len(self.portuguese_contractions),
                "preserved_chars": len(self.preserve_chars["pt"]),
                "dialect_patterns": len(self.dialect_patterns.get("pt", []))
            }
        }

def normalize_text_universal(text: str, language: str = None) -> str:
    """
    Función helper para normalización rápida.
    
    Args:
        text: Texto a normalizar
        language: Idioma. Si None, detecta automáticamente
        
    Returns:
        Texto normalizado
    """
    normalizer = MultilingualNormalizer()
    
    if language:
        return normalizer.normalize_by_language(text, language)
    else:
        normalized_text, _ = normalizer.detect_and_normalize(text)
        return normalized_text