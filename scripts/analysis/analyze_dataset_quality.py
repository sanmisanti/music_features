#!/usr/bin/env python3
"""
Análisis de calidad del dataset real con filtros relajados.
Evalúa el impacto de las correcciones en el success rate.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_quality(dataset_path, max_samples=20):
    """Analiza calidad de letras en dataset real."""
    print(f"📊 ANÁLISIS CALIDAD DATASET: {dataset_path.name}")
    print("=" * 60)
    
    try:
        # Cargar dataset
        print(f"📁 Cargando dataset...")
        df = pd.read_csv(dataset_path, sep='^', encoding='utf-8', nrows=max_samples)
        print(f"   Dataset cargado: {len(df)} filas")
        
        if 'lyrics' not in df.columns:
            print("❌ Columna 'lyrics' no encontrada")
            return
        
        # Importar sistema preprocessing
        from clustering.algorithms.lyrics.preprocessing.feature_extractor import TextFeatureExtractor
        extractor = TextFeatureExtractor()
        
        # Análisis detallado
        results = []
        suitable_count = 0
        
        print(f"\\n🔍 ANALIZANDO {len(df)} CANCIONES...")
        print("-" * 60)
        
        for idx, row in df.iterrows():
            lyrics = str(row.get('lyrics', ''))
            track_name = str(row.get('track_name', f'Track_{idx}'))[:30]
            
            if not lyrics or lyrics == 'nan':
                result = {"suitable": False, "score": 0.0, "issues": ["no_lyrics"]}
            else:
                result = extractor.assess_text_quality(lyrics)
            
            results.append({
                "track": track_name,
                "suitable": result["is_suitable"],
                "score": result["quality_score"],
                "issues": result["issues"],
                "char_count": len(lyrics),
                "ttr": result.get("diversity_ttr", 0.0),
                "repetition": result.get("repetition_ratio", 0.0)
            })
            
            if result["is_suitable"]:
                suitable_count += 1
            
            # Mostrar resultado
            status = "✅" if result["is_suitable"] else "❌"
            print(f"{status} {track_name:30} | Score: {result['quality_score']:.3f} | Issues: {len(result['issues'])}")
        
        # Estadísticas finales
        success_rate = (suitable_count / len(results)) * 100
        
        print("\\n" + "=" * 60)
        print("📈 ESTADÍSTICAS FINALES")
        print("=" * 60)
        print(f"Success Rate: {suitable_count}/{len(results)} ({success_rate:.1f}%)")
        
        # Distribución scores
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"Score promedio: {avg_score:.3f}")
        print(f"Score mínimo: {min(scores):.3f}")
        print(f"Score máximo: {max(scores):.3f}")
        
        # Issues más comunes
        all_issues = []
        for r in results:
            all_issues.extend(r["issues"])
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        
        print("\\nIssues más comunes:")
        for issue, count in issue_counts.most_common(5):
            print(f"  {issue}: {count} ({count/len(results)*100:.1f}%)")
        
        # Distribución características
        ttrs = [r["ttr"] for r in results if r["ttr"] > 0]
        reps = [r["repetition"] for r in results if r["repetition"] > 0]
        
        if ttrs:
            print(f"\\nDiversidad léxica (TTR):")
            print(f"  Promedio: {sum(ttrs)/len(ttrs):.3f}")
            print(f"  Mínimo: {min(ttrs):.3f}")
            print(f"  Máximo: {max(ttrs):.3f}")
        
        if reps:
            print(f"\\nRepetición:")
            print(f"  Promedio: {sum(reps)/len(reps):.3f}")
            print(f"  Mínimo: {min(reps):.3f}")
            print(f"  Máximo: {max(reps):.3f}")
        
        # Recomendaciones
        print("\\n💡 RECOMENDACIONES:")
        if success_rate < 30:
            print("   🔴 Success rate muy bajo - considerar relajar más los filtros")
        elif success_rate < 60:
            print("   🟡 Success rate moderado - filtros balanceados")
        else:
            print("   🟢 Success rate bueno - filtros apropiados")
        
        # Mostrar ejemplos suitable/no suitable
        print("\\n🔍 EJEMPLOS:")
        
        suitable_examples = [r for r in results if r["suitable"]]
        if suitable_examples:
            example = suitable_examples[0]
            print(f"✅ SUITABLE: {example['track']} (score: {example['score']:.3f})")
        
        unsuitable_examples = [r for r in results if not r["suitable"]]
        if unsuitable_examples:
            example = unsuitable_examples[0]
            print(f"❌ NOT SUITABLE: {example['track']} (score: {example['score']:.3f}, issues: {example['issues']})")
        
        return success_rate
        
    except Exception as e:
        print(f"❌ Error analizando dataset: {e}")
        return 0

def main():
    """Ejecuta análisis completo."""
    print("🚀 ANÁLISIS CALIDAD DATASET CON FILTROS RELAJADOS")
    print("=" * 60)
    
    # Ubicaciones posibles dataset
    dataset_paths = [
        project_root / "data" / "3_selected/picked_data_optimal.csv",
        project_root / "data" / "with_lyrics" / "spotify_songs_fixed.csv",
        project_root / "data" / "3_selected/picked_data_optimal.csv"
    ]
    
    found_datasets = []
    for path in dataset_paths:
        if path.exists():
            found_datasets.append(path)
            print(f"✅ Dataset encontrado: {path.name}")
        else:
            print(f"❌ Dataset no existe: {path.name}")
    
    if not found_datasets:
        print("❌ No se encontraron datasets para analizar")
        return
    
    results = {}
    
    # Analizar cada dataset
    for dataset_path in found_datasets:
        print(f"\\n{'='*80}")
        success_rate = analyze_dataset_quality(dataset_path, max_samples=10)
        results[dataset_path.name] = success_rate
    
    # Resumen final
    print(f"\\n{'='*80}")
    print("🏆 RESUMEN FINAL")
    print("="*80)
    
    for dataset_name, success_rate in results.items():
        status = "🟢" if success_rate >= 60 else "🟡" if success_rate >= 30 else "🔴"
        print(f"{status} {dataset_name}: {success_rate:.1f}% success rate")
    
    # Conclusión
    best_rate = max(results.values()) if results else 0
    if best_rate >= 60:
        print("\\n✅ CORRECCIONES EXITOSAS: Success rate mejorado significativamente")
    elif best_rate >= 30:
        print("\\n⚠️ MEJORAS MODERADAS: Considerar ajustes adicionales")
    else:
        print("\\n❌ PROBLEMAS PERSISTEN: Revisar filtros más a fondo")

if __name__ == "__main__":
    main()