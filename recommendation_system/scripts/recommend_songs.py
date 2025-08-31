#!/usr/bin/env python3
"""
Script Principal de Recomendaciones Musicales con Explicabilidad
==============================================================

Sistema completo de recomendaciones musicales con explicaciones automáticas
Integra motor híbrido, análisis de clusters y generación de explicaciones

Uso:
    python recommend_songs.py --track_id "TRACK_ID"
    python recommend_songs.py --song_name "Bohemian Rhapsody" --artist "Queen"
    python recommend_songs.py --interactive
    python recommend_songs.py --demo
    
Autor: Sistema de Clustering Multimodal
Fecha: Agosto 2025
"""

import argparse
import sys
import json
from typing import Dict, List, Optional
from pathlib import Path

# Agregar directorio de scripts al path para importaciones
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from load_system import MusicDataLoader
    from music_recommender import HybridMusicRecommender
    from explain_recommendations import RecommendationExplainer
except ImportError as e:
    print(f"❌ Error importando componentes del sistema: {e}")
    print("   Asegúrate de que todos los scripts estén en el directorio 'scripts/'")
    sys.exit(1)

class MusicRecommendationInterface:
    """
    Interface principal para el sistema de recomendaciones musicales
    Proporciona funcionalidad completa con explicaciones automáticas
    """
    
    def __init__(self, system_dir: str = None, verbose: bool = True, 
                 custom_musical_weight: float = None, custom_semantic_weight: float = None):
        """
        Inicializa el sistema de recomendaciones completo
        
        Args:
            system_dir: Directorio del sistema (None para auto-detectar)
            verbose: Mostrar información detallada de inicialización
            custom_musical_weight: Peso personalizado para componente musical (0.0-1.0)
            custom_semantic_weight: Peso personalizado para componente semántico (0.0-1.0)
        """
        self.verbose = verbose
        
        try:
            if verbose:
                print("🎵 Inicializando Sistema de Recomendaciones Musicales...")
                print("   Cargando componentes...")
            
            # Inicializar componentes principales
            self.loader = MusicDataLoader(system_dir)
            self.recommender = HybridMusicRecommender(
                system_dir, 
                custom_musical_weight=custom_musical_weight,
                custom_semantic_weight=custom_semantic_weight
            )
            self.explainer = RecommendationExplainer(system_dir)
            
            if verbose:
                stats = self.loader.get_system_stats()
                print(f"✅ Sistema inicializado exitosamente")
                print(f"   Dataset: {stats['dataset_size']} canciones")
                print(f"   Clusters musicales: K={stats['cluster_counts']['musical']}")
                print(f"   Clusters semánticos: K={stats['cluster_counts']['semantic']}")
                
        except Exception as e:
            print(f"❌ Error inicializando el sistema: {e}")
            raise
    
    def recommend_by_track_id(self, track_id: str, n_recommendations: int = 10,
                             include_explanations: bool = True, 
                             exclude_same_artist: bool = False) -> Dict:
        """
        Genera recomendaciones por track_id con explicaciones opcionales
        
        Args:
            track_id: ID único de la canción
            n_recommendations: Número de recomendaciones a generar
            include_explanations: Incluir explicaciones automáticas
            exclude_same_artist: Excluir canciones del mismo artista
            
        Returns:
            Dict con recomendaciones y explicaciones
        """
        try:
            # Verificar que la canción existe
            song_info = self.loader.get_song_by_track_id(track_id)
            if not song_info:
                return {
                    'success': False,
                    'error': f'Canción con track_id "{track_id}" no encontrada en el dataset',
                    'suggestions': 'Usa --search para buscar canciones por nombre'
                }
            
            # Generar recomendaciones
            if self.verbose:
                print(f"🎯 Generando {n_recommendations} recomendaciones para:")
                print(f"   {song_info['track_name']} - {song_info['artist_name']}")
                print(f"   Cluster Musical: {song_info['musical_cluster']}")
                print(f"   Cluster Semántico: {song_info['semantic_cluster']}")
            
            recommendations = self.recommender.recommend(
                track_id=track_id,
                n_recommendations=n_recommendations,
                exclude_same_artist=exclude_same_artist
            )
            
            if isinstance(recommendations, list) and len(recommendations) > 0 and 'error' in str(recommendations[0]):
                return {
                    'success': False,
                    'error': recommendations[0].get('error', 'Error desconocido en recomendaciones')
                }
            
            result = {
                'success': True,
                'input_song': recommendations['input_song'],
                'recommendations': recommendations['recommendations'],
                'metadata': recommendations['metadata']
            }
            
            # Agregar explicaciones si se solicitan
            if include_explanations:
                if self.verbose:
                    print("📝 Generando explicaciones automáticas...")
                
                try:
                    explanations = self.explainer.get_batch_explanations(recommendations)
                    
                    if 'error' not in explanations:
                        result['explanations'] = explanations['explanations']
                        result['explanations_metadata'] = {
                            'total_explanations': explanations['total_explanations'],
                            'generation_successful': True
                        }
                    else:
                        result['explanations_metadata'] = {
                            'generation_successful': False,
                            'error': explanations['error']
                        }
                        
                except Exception as e:
                    result['explanations_metadata'] = {
                        'generation_successful': False,
                        'error': f'Error generando explicaciones: {str(e)}'
                    }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en recomendación por track_id: {str(e)}'
            }
    
    def search_songs(self, song_name: str, artist_name: str = None, limit: int = 10) -> List[Dict]:
        """
        Busca canciones por nombre y artista
        
        Args:
            song_name: Nombre de la canción (búsqueda parcial)
            artist_name: Nombre del artista (opcional)
            limit: Máximo número de resultados
            
        Returns:
            Lista de canciones que coinciden
        """
        try:
            # Búsqueda básica por nombre
            matches = self.loader.find_songs_by_name(song_name, limit=limit * 2)  # Buscar más para filtrar
            
            # Filtrar por artista si se especifica
            if artist_name and matches:
                filtered_matches = []
                for song in matches:
                    if artist_name.lower() in song['artist_name'].lower():
                        filtered_matches.append(song)
                matches = filtered_matches[:limit]
            else:
                matches = matches[:limit]
            
            return matches
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error en búsqueda: {e}")
            return []
    
    def recommend_by_name(self, song_name: str, artist_name: str = None,
                         n_recommendations: int = 10, include_explanations: bool = True) -> Dict:
        """
        Genera recomendaciones buscando canción por nombre
        
        Args:
            song_name: Nombre de la canción
            artist_name: Nombre del artista (opcional)
            n_recommendations: Número de recomendaciones
            include_explanations: Incluir explicaciones automáticas
            
        Returns:
            Dict con recomendaciones y explicaciones
        """
        try:
            # Buscar canciones que coincidan
            matches = self.search_songs(song_name, artist_name, limit=10)
            
            if not matches:
                return {
                    'success': False,
                    'error': f'No se encontraron canciones con nombre "{song_name}"' +
                            (f' del artista "{artist_name}"' if artist_name else ''),
                    'suggestions': 'Prueba con una búsqueda más simple o verifica la ortografía'
                }
            
            # Si hay múltiples coincidencias, usar la primera o mostrar opciones
            if len(matches) > 1 and self.verbose:
                print(f"ℹ️  Encontradas {len(matches)} coincidencias:")
                for i, match in enumerate(matches[:5], 1):
                    print(f"   {i}. {match['track_name']} - {match['artist_name']} ({match['genre']})")
                print(f"   Usando la primera coincidencia...")
            
            selected_song = matches[0]
            
            # Generar recomendaciones usando el track_id seleccionado
            return self.recommend_by_track_id(
                track_id=selected_song['track_id'],
                n_recommendations=n_recommendations,
                include_explanations=include_explanations
            )
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error en recomendación por nombre: {str(e)}'
            }
    
    def get_song_analysis(self, track_id: str) -> Dict:
        """
        Obtiene análisis completo de clusters para una canción específica
        
        Args:
            track_id: ID de la canción
            
        Returns:
            Dict con análisis de clusters musicales y semánticos
        """
        try:
            song_info = self.loader.get_song_by_track_id(track_id)
            if not song_info:
                return {'error': f'Canción {track_id} no encontrada'}
            
            musical_analysis = self.explainer.analyze_musical_cluster(song_info['musical_cluster'])
            semantic_analysis = self.explainer.analyze_semantic_cluster(song_info['semantic_cluster'])
            
            return {
                'song_info': song_info,
                'musical_cluster_analysis': musical_analysis,
                'semantic_cluster_analysis': semantic_analysis
            }
            
        except Exception as e:
            return {'error': f'Error en análisis: {str(e)}'}


def format_recommendations_output(result: Dict, show_explanations: bool = True, 
                                compact: bool = False) -> str:
    """
    Formatea la salida de recomendaciones para visualización
    
    Args:
        result: Resultado del sistema de recomendaciones
        show_explanations: Mostrar explicaciones detalladas
        compact: Formato compacto o detallado
        
    Returns:
        String formateado para mostrar al usuario
    """
    if not result['success']:
        return f"❌ Error: {result['error']}\n" + (result.get('suggestions', '') and f"💡 {result['suggestions']}\n")
    
    output_lines = []
    
    # Información de la canción de entrada
    input_song = result['input_song']
    output_lines.append("🎵 CANCIÓN DE ENTRADA:")
    output_lines.append(f"   {input_song['track_name']} - {input_song['artist_name']}")
    output_lines.append(f"   Género: {input_song['genre']}")
    output_lines.append(f"   Cluster Musical: {input_song['musical_cluster']} | Cluster Semántico: {input_song['semantic_cluster']}")
    output_lines.append("")
    
    # Recomendaciones
    recommendations = result['recommendations']
    metadata = result['metadata']
    
    output_lines.append(f"🎯 RECOMENDACIONES ({len(recommendations)} generadas en {metadata['execution_time_ms']:.1f}ms):")
    output_lines.append("")
    
    for i, rec in enumerate(recommendations, 1):
        output_lines.append(f"{i}. {rec['track_name']} - {rec['artist_name']}")
        if not compact:
            # Métricas básicas
            output_lines.append(f"   Género: {rec['genre']} | Similitud Híbrida: {rec['scores']['hybrid']:.3f}")
            output_lines.append(f"   Musical: {rec['musical_cluster']} | Semántico: {rec['semantic_cluster']}")
            
            # Métricas expandidas
            scores = rec['scores']
            output_lines.append(f"   📊 Similitudes: Musical {scores['musical']:.3f} | Semántico {scores['semantic']:.3f}")
            output_lines.append(f"   🎯 Contribuciones: Musical {scores['musical_contribution']:.3f} | Semántico {scores['semantic_contribution']:.3f}")
            
            # Características musicales
            if 'musical_features' in rec:
                features = rec['musical_features']
                output_lines.append(f"   🎵 Características: Energía {features['energy']:.2f} | Baile {features['danceability']:.2f} | Valencia {features['valence']:.2f}")
                output_lines.append(f"   🎼 Audio: Acústico {features['acousticness']:.2f} | Instrumental {features['instrumentalness']:.2f} | Tempo {features['tempo']:.0f}")
            
            # Información adicional
            if 'additional_info' in rec:
                info = rec['additional_info']
                duration_min = int(info.get('duration_ms', 0)) // 60000
                duration_sec = (int(info.get('duration_ms', 0)) % 60000) // 1000
                output_lines.append(f"   ℹ️  Album: {info.get('album_name', 'Unknown')} ({info.get('release_date', 'Unknown')})")
                output_lines.append(f"   ⏱️  Duración: {duration_min}:{duration_sec:02d} | Popularidad: {info.get('popularity', 0)}")
        else:
            output_lines.append(f"   Género: {rec['genre']} | Similitud: {rec['scores']['hybrid']:.3f}")
        
        # Agregar explicación si está disponible
        if show_explanations and 'explanations' in result:
            explanations = result['explanations']
            if i <= len(explanations) and 'error' not in explanations[i-1]:
                exp = explanations[i-1]['explanation']
                if not compact:
                    output_lines.append(f"   📝 {exp['full_explanation']}")
                else:
                    output_lines.append(f"   📝 {exp['main_reason']}")
        
        output_lines.append("")
    
    # Metadatos adicionales
    if not compact:
        output_lines.append(f"⚡ Performance: {metadata['execution_time_ms']:.1f}ms")
        output_lines.append(f"🎛️  Pesos: Musical {metadata['weights_used']['musical']:.2f} | Semántico {metadata['weights_used']['semantic']:.2f}")
        
        if 'explanations_metadata' in result:
            exp_meta = result['explanations_metadata']
            if exp_meta['generation_successful']:
                output_lines.append(f"📝 Explicaciones: {exp_meta['total_explanations']} generadas")
            else:
                output_lines.append(f"⚠️ Explicaciones: {exp_meta.get('error', 'Error desconocido')}")
    
    return "\n".join(output_lines)


def interactive_mode(interface: MusicRecommendationInterface):
    """
    Modo interactivo para recomendaciones musicales
    
    Args:
        interface: Interface del sistema de recomendaciones
    """
    print("🎵 Modo Interactivo - Sistema de Recomendaciones Musicales")
    print("   Comandos disponibles:")
    print("   - 'search <nombre_cancion>' : Buscar canciones")
    print("   - 'recommend <track_id>' : Recomendar por ID")
    print("   - 'analyze <track_id>' : Analizar clusters de una canción")
    print("   - 'demo' : Ejecutar demostración")
    print("   - 'help' : Mostrar ayuda")
    print("   - 'quit' : Salir")
    print("")
    
    while True:
        try:
            command = input("🎯 Comando: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            elif command.lower() == 'help':
                print("\n📖 Ayuda:")
                print("   search bohemian rhapsody     - Buscar canciones por nombre")
                print("   recommend TRACK_ID           - Generar recomendaciones")
                print("   analyze TRACK_ID             - Análisis de clusters")
                print("   demo                         - Demostración completa")
                print("")
                continue
            
            elif command.lower() == 'demo':
                run_demo(interface)
                continue
            
            elif command.startswith('search '):
                query = command[7:].strip()
                if not query:
                    print("❌ Especifica el nombre de la canción a buscar")
                    continue
                
                results = interface.search_songs(query, limit=10)
                
                if results:
                    print(f"\n🔍 Encontradas {len(results)} coincidencias:")
                    for i, song in enumerate(results, 1):
                        print(f"   {i}. {song['track_name']} - {song['artist_name']}")
                        print(f"      ID: {song['track_id']} | Género: {song['genre']}")
                    print("")
                else:
                    print(f"❌ No se encontraron canciones con '{query}'")
                
            elif command.startswith('recommend '):
                track_id = command[10:].strip()
                if not track_id:
                    print("❌ Especifica el track_id")
                    continue
                
                print(f"\n⏳ Generando recomendaciones para {track_id}...")
                result = interface.recommend_by_track_id(track_id, n_recommendations=5)
                
                print(format_recommendations_output(result, show_explanations=True, compact=False))
                
            elif command.startswith('analyze '):
                track_id = command[8:].strip()
                if not track_id:
                    print("❌ Especifica el track_id")
                    continue
                
                print(f"\n🔍 Analizando clusters para {track_id}...")
                analysis = interface.get_song_analysis(track_id)
                
                if 'error' in analysis:
                    print(f"❌ {analysis['error']}")
                else:
                    song = analysis['song_info']
                    musical = analysis['musical_cluster_analysis']
                    semantic = analysis['semantic_cluster_analysis']
                    
                    print(f"\n🎵 {song['track_name']} - {song['artist_name']}")
                    print(f"🎶 Cluster Musical {song['musical_cluster']}: {musical['cluster_label']}")
                    print(f"   {musical['interpretation']}")
                    print(f"🧠 Cluster Semántico {song['semantic_cluster']}: {semantic['cluster_label']}")
                    print(f"   {semantic['interpretation']}")
                    print("")
                
            else:
                print("❌ Comando no reconocido. Usa 'help' para ver opciones disponibles.")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_demo(interface: MusicRecommendationInterface):
    """
    Ejecuta demostración completa del sistema
    
    Args:
        interface: Interface del sistema de recomendaciones
    """
    print("\n🎭 DEMOSTRACIÓN DEL SISTEMA DE RECOMENDACIONES MUSICALES")
    print("=" * 60)
    
    try:
        # Obtener una canción de muestra
        track_ids = interface.loader.get_track_ids()
        if len(track_ids) == 0:
            print("❌ No hay canciones disponibles para demostración")
            return
        
        demo_track_id = track_ids[0]
        song_info = interface.loader.get_song_by_track_id(demo_track_id)
        
        print(f"\n📋 PASO 1: Análisis de canción de muestra")
        print(f"   Canción: {song_info['track_name']} - {song_info['artist_name']}")
        print(f"   Género: {song_info['genre']}")
        
        # Análisis de clusters
        analysis = interface.get_song_analysis(demo_track_id)
        if 'error' not in analysis:
            musical = analysis['musical_cluster_analysis']
            semantic = analysis['semantic_cluster_analysis']
            print(f"   Cluster Musical: {musical['cluster_label']}")
            print(f"   Cluster Semántico: {semantic['cluster_label']}")
        
        print(f"\n📋 PASO 2: Generación de recomendaciones")
        result = interface.recommend_by_track_id(demo_track_id, n_recommendations=3, include_explanations=True)
        
        if result['success']:
            print(f"   ✅ {len(result['recommendations'])} recomendaciones generadas en {result['metadata']['execution_time_ms']:.1f}ms")
            
            print(f"\n📋 PASO 3: Visualización con explicaciones")
            print(format_recommendations_output(result, show_explanations=True, compact=False))
            
            print(f"\n📋 PASO 4: Métricas de performance")
            metadata = result['metadata']
            print(f"   ⚡ Tiempo de ejecución: {metadata['execution_time_ms']:.1f}ms (Target <100ms)")
            print(f"   🎯 Performance target: {'✅ Cumplido' if metadata['execution_time_ms'] < 100 else '⚠️ No cumplido'}")
            print(f"   🎛️  Configuración híbrida: {metadata['weights_used']['musical']:.0%} musical, {metadata['weights_used']['semantic']:.0%} semántica")
            
        else:
            print(f"   ❌ Error en demostración: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error en demostración: {e}")
    
    print("\n🏆 Demostración completada")


def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendaciones Musicales con Explicabilidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python recommend_songs.py --track_id "4uLU6hMCjMI75M1A2tKUQC"
  python recommend_songs.py --song_name "Bohemian Rhapsody" --artist "Queen"  
  python recommend_songs.py --search "stairway to heaven"
  python recommend_songs.py --interactive
  python recommend_songs.py --demo
        """
    )
    
    # Argumentos principales
    parser.add_argument('--track_id', type=str, help='ID único de la canción')
    parser.add_argument('--song_name', type=str, help='Nombre de la canción')
    parser.add_argument('--artist', type=str, help='Nombre del artista (opcional)')
    parser.add_argument('--search', type=str, help='Buscar canciones por nombre')
    
    # Opciones de recomendación
    parser.add_argument('--n_recommendations', type=int, default=10, 
                       help='Número de recomendaciones (default: 10)')
    parser.add_argument('--no_explanations', action='store_true',
                       help='No incluir explicaciones automáticas')
    parser.add_argument('--exclude_same_artist', action='store_true',
                       help='Excluir canciones del mismo artista')
    
    # Modos especiales
    parser.add_argument('--interactive', action='store_true',
                       help='Modo interactivo')
    parser.add_argument('--demo', action='store_true',
                       help='Ejecutar demostración completa')
    parser.add_argument('--analyze', type=str, 
                       help='Analizar clusters de una canción por track_id')
    
    # Opciones de formato
    parser.add_argument('--compact', action='store_true',
                       help='Formato de salida compacto')
    parser.add_argument('--json_output', action='store_true',
                       help='Salida en formato JSON')
    parser.add_argument('--quiet', action='store_true',
                       help='Modo silencioso (solo resultados)')
    
    # Opciones técnicas
    parser.add_argument('--system_dir', type=str,
                       help='Directorio del sistema (para desarrollo)')
    
    # Opciones de personalización de pesos
    parser.add_argument('--musical_weight', type=float,
                       help='Peso para componente musical (0.0-1.0, debe sumar 1.0 con --semantic_weight)')
    parser.add_argument('--semantic_weight', type=float,
                       help='Peso para componente semántico (0.0-1.0, debe sumar 1.0 con --musical_weight)')
    
    args = parser.parse_args()
    
    try:
        # Inicializar sistema
        verbose = not args.quiet
        
        # Manejar pesos personalizados
        custom_musical_weight = args.musical_weight
        custom_semantic_weight = args.semantic_weight
        
        # Validar pesos si se proporcionan
        if (custom_musical_weight is not None) != (custom_semantic_weight is not None):
            print("❌ Error: Debe especificar tanto --musical_weight como --semantic_weight juntos")
            return
        
        if custom_musical_weight is not None and custom_semantic_weight is not None:
            if not (0.0 <= custom_musical_weight <= 1.0) or not (0.0 <= custom_semantic_weight <= 1.0):
                print("❌ Error: Los pesos deben estar en rango 0.0-1.0")
                return
            
            if abs((custom_musical_weight + custom_semantic_weight) - 1.0) > 0.001:
                print(f"❌ Error: Los pesos deben sumar 1.0 (suma actual: {custom_musical_weight + custom_semantic_weight:.3f})")
                return
        
        interface = MusicRecommendationInterface(
            args.system_dir, 
            verbose=verbose,
            custom_musical_weight=custom_musical_weight,
            custom_semantic_weight=custom_semantic_weight
        )
        
        # Modo interactivo
        if args.interactive:
            interactive_mode(interface)
            return
        
        # Modo demostración
        if args.demo:
            run_demo(interface)
            return
        
        # Análisis de canción
        if args.analyze:
            analysis = interface.get_song_analysis(args.analyze)
            
            if args.json_output:
                print(json.dumps(analysis, indent=2, ensure_ascii=False))
            elif 'error' in analysis:
                print(f"❌ Error: {analysis['error']}")
            else:
                song = analysis['song_info']
                musical = analysis['musical_cluster_analysis']
                semantic = analysis['semantic_cluster_analysis']
                
                print(f"🎵 ANÁLISIS DE {song['track_name']} - {song['artist_name']}")
                print(f"🎶 Cluster Musical {song['musical_cluster']}: {musical['cluster_label']}")
                print(f"   {musical['interpretation']}")
                print(f"🧠 Cluster Semántico {song['semantic_cluster']}: {semantic['cluster_label']}")
                print(f"   {semantic['interpretation']}")
            return
        
        # Búsqueda de canciones
        if args.search:
            results = interface.search_songs(args.search, limit=15)
            
            if args.json_output:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            elif results:
                print(f"🔍 RESULTADOS DE BÚSQUEDA PARA '{args.search}':")
                for i, song in enumerate(results, 1):
                    print(f"{i:2d}. {song['track_name']} - {song['artist_name']}")
                    print(f"     ID: {song['track_id']} | Género: {song['genre']}")
            else:
                print(f"❌ No se encontraron canciones con '{args.search}'")
            return
        
        # Recomendaciones por track_id
        if args.track_id:
            result = interface.recommend_by_track_id(
                track_id=args.track_id,
                n_recommendations=args.n_recommendations,
                include_explanations=not args.no_explanations,
                exclude_same_artist=args.exclude_same_artist
            )
            
            if args.json_output:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(format_recommendations_output(
                    result, 
                    show_explanations=not args.no_explanations,
                    compact=args.compact
                ))
            return
        
        # Recomendaciones por nombre de canción
        if args.song_name:
            result = interface.recommend_by_name(
                song_name=args.song_name,
                artist_name=args.artist,
                n_recommendations=args.n_recommendations,
                include_explanations=not args.no_explanations
            )
            
            if args.json_output:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(format_recommendations_output(
                    result,
                    show_explanations=not args.no_explanations,
                    compact=args.compact
                ))
            return
        
        # Si no se especifica ningún argumento, mostrar ayuda
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()