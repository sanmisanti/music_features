"""
Gestión de stopwords especializadas para letras musicales.
Combina stopwords tradicionales + específicas musicales por idioma.

Características:
- Stopwords tradicionales NLTK por idioma
- Stopwords musicales personalizadas  
- Gestión interjecciones musicales
- Filtrado contextual inteligente
"""

import os
import logging
from typing import Set, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class StopwordsManager:
    """
    Gestor especializado stopwords para letras musicales multilingües.
    
    Combina:
    - Stopwords estándar NLTK
    - Stopwords musicales personalizadas
    - Interjecciones musicales por idioma
    - Filtrado contextual inteligente
    """
    
    def __init__(self, custom_stopwords_dir: Optional[Path] = None):
        """
        Inicializa gestor stopwords.
        
        Args:
            custom_stopwords_dir: Directorio stopwords personalizadas.
                                 Si None, usa directorio por defecto del módulo.
        """
        self.custom_stopwords_dir = custom_stopwords_dir or self._get_default_stopwords_dir()
        self.stopwords_cache = {}
        self.nltk_available = False
        
        self._initialize_nltk()
        self._load_custom_stopwords()
        
        logger.debug("StopwordsManager inicializado")
    
    def _get_default_stopwords_dir(self) -> Path:
        """Obtiene directorio por defecto stopwords personalizadas."""
        current_file = Path(__file__)
        lyrics_module = current_file.parent.parent
        return lyrics_module / "data" / "stopwords"
    
    def _initialize_nltk(self):
        """Inicializa stopwords NLTK si están disponibles."""
        try:
            import nltk
            from nltk.corpus import stopwords
            
            # Verificar que stopwords estén descargadas
            try:
                stopwords.words('english')
                self.nltk_available = True
                logger.debug("NLTK stopwords disponibles")
            except LookupError:
                logger.warning("NLTK stopwords no descargadas. Usando solo stopwords personalizadas")
                
        except ImportError:
            logger.warning("NLTK no disponible. Usando solo stopwords personalizadas")
    
    def _load_custom_stopwords(self):
        """Carga stopwords personalizadas desde archivos."""
        
        # Stopwords musicales base (hardcoded como fallback)
        self.base_music_stopwords = {
            "en": {
                # Interjecciones musicales comunes
                "yeah", "oh", "ah", "eh", "uh", "mm", "hmm", "hey", "yo", "whoa", 
                "ooh", "aah", "la", "da", "na", "ba", "sha", "doo", "woo", "tra", 
                "ho", "ha", "ay", "ey", "oy",
                
                # Estructuras repetitivas
                "chorus", "verse", "bridge", "outro", "intro", "repeat", "x2", "x3", "x4",
                
                # Slang musical común  
                "baby", "come", "go", "get", "got", "make", "take", "give", "know",
                "love", "like", "want", "need", "feel", "look", "see", "say", "tell"
            },
            
            "es": {
                # Interjecciones
                "ay", "eh", "oh", "uy", "ah", "mm", "hey", "ya", "na", "la", "oh",
                
                # Estructuras
                "estribillo", "verso", "coro", "final", "intro",
                
                # Comunes musicales
                "amor", "vida", "corazón", "alma", "sueño", "tiempo", "noche", "día",
                "dame", "dime", "ven", "vamos", "quiero", "tengo", "soy", "estoy"
            },
            
            "de": {
                # Interjecciones
                "oh", "ja", "eh", "na", "so", "hey", "ah", "mm",
                
                # Estructuras
                "refrain", "strophe", "bridge", "outro", "intro",
                
                # Comunes musicales
                "liebe", "herz", "leben", "traum", "nacht", "tag", "komm", "geh",
                "bin", "bist", "ist", "sind", "hab", "hast", "hat", "haben"
            },
            
            "pt": {
                # Interjecciones
                "ai", "eh", "né", "oh", "já", "ah", "mm", "hey", "na", "la",
                
                # Estructuras
                "refrão", "verso", "ponte", "final", "intro",
                
                # Comunes musicales
                "amor", "vida", "coração", "alma", "sonho", "tempo", "noite", "dia",
                "vem", "vai", "fica", "sou", "és", "é", "somos", "são", "tenho", "tens"
            }
        }
        
        # Intentar cargar stopwords personalizadas de archivos
        for language in ["english", "spanish", "german", "portuguese"]:
            lang_code = self._get_language_code(language)
            file_path = self.custom_stopwords_dir / f"{language}.txt"
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        custom_words = {line.strip().lower() for line in f if line.strip()}
                    
                    # Combinar con stopwords base
                    if lang_code in self.base_music_stopwords:
                        self.base_music_stopwords[lang_code].update(custom_words)
                    else:
                        self.base_music_stopwords[lang_code] = custom_words
                    
                    logger.debug(f"Stopwords personalizadas cargadas para {language}: {len(custom_words)} palabras")
                    
                except Exception as e:
                    logger.warning(f"Error cargando stopwords personalizadas {language}: {e}")
            else:
                logger.debug(f"Archivo stopwords {language} no encontrado, usando base")
    
    def _get_language_code(self, language_name: str) -> str:
        """Convierte nombre idioma a código."""
        mapping = {
            "english": "en",
            "spanish": "es", 
            "german": "de",
            "portuguese": "pt"
        }
        return mapping.get(language_name.lower(), language_name.lower())
    
    def get_stopwords(self, language: str, include_nltk: bool = True, 
                      include_musical: bool = True, include_contextual: bool = False) -> Set[str]:
        """
        Obtiene stopwords para idioma específico.
        
        Args:
            language: Código idioma (en, es, de, pt)
            include_nltk: Incluir stopwords NLTK estándar
            include_musical: Incluir stopwords musicales personalizadas
            include_contextual: Incluir stopwords contextuales adicionales
            
        Returns:
            Set de stopwords para el idioma
        """
        cache_key = f"{language}_{include_nltk}_{include_musical}_{include_contextual}"
        
        if cache_key in self.stopwords_cache:
            return self.stopwords_cache[cache_key]
        
        stopwords_set = set()
        
        # 1. Stopwords NLTK estándar
        if include_nltk and self.nltk_available:
            try:
                from nltk.corpus import stopwords
                
                # Mapear códigos a nombres NLTK
                nltk_names = {"en": "english", "es": "spanish", "de": "german", "pt": "portuguese"}
                nltk_name = nltk_names.get(language)
                
                if nltk_name:
                    nltk_stopwords = set(stopwords.words(nltk_name))
                    stopwords_set.update(nltk_stopwords)
                    logger.debug(f"NLTK stopwords {language}: {len(nltk_stopwords)} palabras")
                    
            except Exception as e:
                logger.warning(f"Error cargando NLTK stopwords {language}: {e}")
        
        # 2. Stopwords musicales personalizadas
        if include_musical:
            musical_stopwords = self.base_music_stopwords.get(language, set())
            stopwords_set.update(musical_stopwords)
            logger.debug(f"Stopwords musicales {language}: {len(musical_stopwords)} palabras")
        
        # 3. Stopwords contextuales adicionales
        if include_contextual:
            contextual_stopwords = self._get_contextual_stopwords(language)
            stopwords_set.update(contextual_stopwords)
            logger.debug(f"Stopwords contextuales {language}: {len(contextual_stopwords)} palabras")
        
        # Cache resultado
        self.stopwords_cache[cache_key] = stopwords_set
        
        logger.debug(f"Stopwords totales {language}: {len(stopwords_set)} palabras")
        return stopwords_set
    
    def _get_contextual_stopwords(self, language: str) -> Set[str]:
        """
        Obtiene stopwords contextuales adicionales por idioma.
        
        Estas son palabras que pueden ser útiles en algunos contextos
        pero problemáticas en análisis semántico musical.
        """
        contextual_stopwords = {
            "en": {
                # Palabras muy comunes que añaden poco valor semántico
                "thing", "things", "way", "ways", "time", "times", "people", "person",
                "world", "life", "day", "days", "night", "nights", "place", "places",
                "hand", "hands", "eye", "eyes", "heart", "mind", "soul", "body",
                
                # Conectores y transiciones
                "now", "then", "here", "there", "where", "when", "how", "why", "what",
                "just", "only", "still", "even", "also", "too", "well", "right", "left"
            },
            
            "es": {
                # Palabras muy comunes
                "cosa", "cosas", "manera", "forma", "vez", "veces", "gente", "persona",
                "mundo", "lugar", "lugares", "mano", "manos", "ojo", "ojos", 
                
                # Conectores
                "ahora", "entonces", "aquí", "allí", "donde", "cuando", "como", "por", "que",
                "solo", "sólo", "aún", "también", "bien", "mal", "si", "sino"
            },
            
            "de": {
                # Palabras muy comunes
                "ding", "dinge", "weg", "wege", "zeit", "zeiten", "leute", "person",
                "welt", "ort", "orte", "hand", "hände", "auge", "augen",
                
                # Conectores
                "jetzt", "dann", "hier", "dort", "wo", "wann", "wie", "warum", "was",
                "nur", "noch", "auch", "gut", "schlecht", "rechts", "links"
            },
            
            "pt": {
                # Palabras muy comunes
                "coisa", "coisas", "maneira", "forma", "vez", "vezes", "gente", "pessoa",
                "mundo", "lugar", "lugares", "mão", "mãos", "olho", "olhos",
                
                # Conectores
                "agora", "então", "aqui", "ali", "onde", "quando", "como", "por", "que",
                "só", "apenas", "ainda", "também", "bem", "mal", "se", "mas"
            }
        }
        
        return contextual_stopwords.get(language, set())
    
    def filter_stopwords(self, words: List[str], language: str, 
                        aggressive: bool = False) -> List[str]:
        """
        Filtra stopwords de una lista de palabras.
        
        Args:
            words: Lista palabras a filtrar
            language: Código idioma
            aggressive: Si True, usa filtrado más agresivo (incluye contextuales)
            
        Returns:
            Lista palabras filtradas
        """
        if not words:
            return []
        
        # Obtener stopwords según nivel agresividad
        stopwords_set = self.get_stopwords(
            language,
            include_nltk=True,
            include_musical=True,
            include_contextual=aggressive
        )
        
        # Filtrar palabras (case-insensitive)
        filtered_words = [word for word in words 
                         if word.lower() not in stopwords_set and len(word) > 1]
        
        reduction_pct = ((len(words) - len(filtered_words)) / len(words)) * 100
        logger.debug(f"Stopwords filtradas {language}: {len(words)} -> {len(filtered_words)} "
                    f"({reduction_pct:.1f}% reducción)")
        
        return filtered_words
    
    def filter_text(self, text: str, language: str, aggressive: bool = False) -> str:
        """
        Filtra stopwords de un texto completo.
        
        Args:
            text: Texto a filtrar
            language: Código idioma
            aggressive: Nivel agresividad filtrado
            
        Returns:
            Texto con stopwords filtradas
        """
        if not text:
            return ""
        
        # Tokenizar simple (split por espacios)
        words = text.split()
        
        # Filtrar stopwords
        filtered_words = self.filter_stopwords(words, language, aggressive)
        
        # Reconstruir texto
        return " ".join(filtered_words)
    
    def is_stopword(self, word: str, language: str, aggressive: bool = False) -> bool:
        """
        Verifica si una palabra es stopword.
        
        Args:
            word: Palabra a verificar
            language: Código idioma
            aggressive: Incluir stopwords contextuales
            
        Returns:
            True si es stopword, False otherwise
        """
        stopwords_set = self.get_stopwords(
            language,
            include_nltk=True,
            include_musical=True,
            include_contextual=aggressive
        )
        
        return word.lower() in stopwords_set
    
    def get_stopwords_stats(self) -> Dict[str, Dict]:
        """Obtiene estadísticas stopwords por idioma."""
        stats = {}
        
        for language in ["en", "es", "de", "pt"]:
            nltk_stopwords = self.get_stopwords(language, include_nltk=True, include_musical=False, include_contextual=False)
            musical_stopwords = self.get_stopwords(language, include_nltk=False, include_musical=True, include_contextual=False)
            contextual_stopwords = self.get_stopwords(language, include_nltk=False, include_musical=False, include_contextual=True)
            total_stopwords = self.get_stopwords(language, include_nltk=True, include_musical=True, include_contextual=True)
            
            stats[language] = {
                "nltk_count": len(nltk_stopwords),
                "musical_count": len(musical_stopwords),
                "contextual_count": len(contextual_stopwords),
                "total_count": len(total_stopwords),
                "coverage_overlap": len(nltk_stopwords & musical_stopwords)
            }
        
        return stats
    
    def add_custom_stopwords(self, language: str, words: List[str]):
        """
        Añade stopwords personalizadas temporalmente.
        
        Args:
            language: Código idioma
            words: Lista palabras a añadir como stopwords
        """
        if language not in self.base_music_stopwords:
            self.base_music_stopwords[language] = set()
        
        self.base_music_stopwords[language].update(word.lower() for word in words)
        
        # Limpiar cache para forzar recarga
        keys_to_remove = [key for key in self.stopwords_cache.keys() if key.startswith(language)]
        for key in keys_to_remove:
            del self.stopwords_cache[key]
        
        logger.debug(f"Añadidas {len(words)} stopwords personalizadas para {language}")
    
    def get_musical_interjections(self, language: str) -> Set[str]:
        """
        Obtiene solo interjecciones musicales para idioma específico.
        
        Args:
            language: Código idioma
            
        Returns:
            Set interjecciones musicales
        """
        all_stopwords = self.base_music_stopwords.get(language, set())
        
        # Filtrar solo interjecciones (palabras cortas, generalmente <= 3 chars)
        interjections = {word for word in all_stopwords if len(word) <= 3}
        
        return interjections

# Helper function para uso rápido
def get_stopwords_for_language(language: str, include_musical: bool = True) -> Set[str]:
    """
    Función helper para obtener stopwords rápidamente.
    
    Args:
        language: Código idioma (en, es, de, pt)
        include_musical: Incluir stopwords musicales
        
    Returns:
        Set stopwords para el idioma
    """
    manager = StopwordsManager()
    return manager.get_stopwords(language, include_musical=include_musical)