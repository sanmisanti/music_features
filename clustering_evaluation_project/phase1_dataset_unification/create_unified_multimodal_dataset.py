#!/usr/bin/env python3
"""
FASE 1: CONSTRUCCIÓN DE DATASET MULTIMODAL UNIFICADO
===================================================

Script para construir dataset unificado con embeddings BERT y características musicales
alineadas para las mismas canciones, con normalización apropiada.

Autor: Clustering Evaluation Project
Fecha: Agosto 2025
Objetivo: Crear base de datos coherente para análisis comparativo multimodal
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class UnifiedMultimodalDatasetBuilder:
    """
    Constructor de dataset multimodal unificado.
    
    Funcionalidades:
    - Alineación de embeddings BERT con características musicales
    - Normalización apropiada de características musicales
    - Validación de integridad de datos multimodales
    - Construcción de estructura optimizada para análisis
    """
    
    def __init__(self, base_path=None):
        """Inicializar constructor de dataset unificado."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rutas de archivos principales
        self.semantic_embeddings_path = base_path / "vectorization_complete_output" / "embeddings_complete_20250819_194820.npy"
        self.semantic_track_ids_path = base_path / "vectorization_complete_output" / "track_ids_complete_20250819_194820.npy"
        self.musical_dataset_path = base_path / "data" / "final_data" / "picked_data_optimal.csv"
        
        # Buscar el archivo de track IDs válidos más reciente
        audit_files = list(Path(__file__).parent.glob("valid_multimodal_track_ids_*.npy"))
        if audit_files:
            self.valid_track_ids_path = sorted(audit_files)[-1]  # Más reciente
            print(f"Usando track IDs válidos de: {self.valid_track_ids_path.name}")
        else:
            raise FileNotFoundError("No se encontró archivo de track IDs válidos. Ejecutar primero dataset_intersection_audit.py")
        
        # Características musicales a utilizar
        self.musical_feature_columns = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        # Estructura de dataset unificado
        self.unified_dataset = {}
        
        print(f"UnifiedMultimodalDatasetBuilder inicializado")
        print(f"Base path: {base_path}")
        print(f"Timestamp: {self.timestamp}")
    
    def load_multimodal_track_ids(self):
        """Cargar lista de track IDs válidos para análisis multimodal."""
        try:
            print("Cargando track IDs multimodales válidos...")
            
            self.valid_multimodal_track_ids = np.load(self.valid_track_ids_path, allow_pickle=True)
            print(f"Track IDs multimodales cargados: {len(self.valid_multimodal_track_ids)}")
            
            # Convertir a set para búsquedas eficientes
            self.valid_track_ids_set = set(self.valid_multimodal_track_ids)
            
            return True
            
        except Exception as e:
            print(f"Error cargando track IDs multimodales: {e}")
            return False
    
    def load_and_filter_semantic_data(self):
        """Cargar y filtrar embeddings semánticos a la intersección multimodal."""
        try:
            print("Cargando y filtrando datos semánticos...")
            
            # Cargar todos los embeddings y track IDs
            all_embeddings = np.load(self.semantic_embeddings_path)
            all_track_ids = np.load(self.semantic_track_ids_path, allow_pickle=True)
            
            print(f"Embeddings totales cargados: {all_embeddings.shape}")
            print(f"Track IDs totales: {len(all_track_ids)}")
            
            # Crear mapeo de track_id a índice
            track_id_to_index = {track_id: idx for idx, track_id in enumerate(all_track_ids)}
            
            # Filtrar a embeddings válidos (no-zero)
            valid_mask = np.any(all_embeddings != 0, axis=1)
            valid_embeddings = all_embeddings[valid_mask]
            valid_track_ids = all_track_ids[valid_mask]
            
            print(f"Embeddings válidos: {len(valid_track_ids)}")
            
            # Filtrar a intersección multimodal
            multimodal_indices = []
            multimodal_track_ids = []
            multimodal_embeddings = []
            
            for track_id in self.valid_multimodal_track_ids:
                # Buscar en embeddings válidos
                valid_indices = np.where(valid_track_ids == track_id)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[0]
                    multimodal_indices.append(idx)
                    multimodal_track_ids.append(track_id)
                    multimodal_embeddings.append(valid_embeddings[idx])
            
            self.semantic_embeddings_filtered = np.array(multimodal_embeddings)
            self.semantic_track_ids_filtered = np.array(multimodal_track_ids)
            
            print(f"Embeddings semánticos filtrados: {self.semantic_embeddings_filtered.shape}")
            
            # Validar dimensionalidad
            if self.semantic_embeddings_filtered.shape[1] != 384:
                raise ValueError(f"Dimensionalidad inesperada: {self.semantic_embeddings_filtered.shape[1]} != 384")
            
            return True
            
        except Exception as e:
            print(f"Error cargando datos semánticos: {e}")
            return False
    
    def load_and_filter_musical_data(self):
        """Cargar y filtrar características musicales a la intersección multimodal."""
        try:
            print("Cargando y filtrando datos musicales...")
            
            # Cargar dataset musical completo
            musical_dataset = pd.read_csv(self.musical_dataset_path, 
                                        sep='^', 
                                        encoding='utf-8',
                                        low_memory=False)
            
            print(f"Dataset musical total: {len(musical_dataset)}")
            
            # Filtrar a intersección multimodal
            musical_filtered = musical_dataset[
                musical_dataset['track_id'].isin(self.valid_track_ids_set)
            ].copy()
            
            print(f"Datos musicales filtrados: {len(musical_filtered)}")
            
            # Verificar disponibilidad de características
            available_features = [col for col in self.musical_feature_columns if col in musical_filtered.columns]
            missing_features = [col for col in self.musical_feature_columns if col not in musical_filtered.columns]
            
            print(f"Características disponibles: {len(available_features)}/12")
            if missing_features:
                print(f"Características faltantes: {missing_features}")
            
            self.available_musical_features = available_features
            self.musical_dataset_filtered = musical_filtered
            
            return True
            
        except Exception as e:
            print(f"Error cargando datos musicales: {e}")
            return False
    
    def align_datasets(self):
        """Alinear datasets semántico y musical por track_id."""
        try:
            print("Alineando datasets por track_id...")
            
            # Verificar duplicados en datasets fuente
            semantic_duplicates = len(self.semantic_track_ids_filtered) - len(set(self.semantic_track_ids_filtered))
            musical_duplicates = len(self.musical_dataset_filtered) - len(self.musical_dataset_filtered['track_id'].unique())
            
            print(f"Duplicados detectados:")
            print(f"  Semánticos: {semantic_duplicates}")
            print(f"  Musicales: {musical_duplicates}")
            
            # Eliminar duplicados de datasets fuente
            if semantic_duplicates > 0:
                print("Eliminando duplicados semánticos...")
                semantic_df = pd.DataFrame({
                    'track_id': self.semantic_track_ids_filtered,
                    'embedding_index': range(len(self.semantic_track_ids_filtered))
                })
                # Mantener primer occurrence de cada track_id
                semantic_df = semantic_df.drop_duplicates(subset=['track_id'], keep='first')
                
                # Actualizar embeddings filtrados
                unique_indices = semantic_df['embedding_index'].values
                self.semantic_embeddings_filtered = self.semantic_embeddings_filtered[unique_indices]
                self.semantic_track_ids_filtered = semantic_df['track_id'].values
                
                print(f"Embeddings únicos después de deduplicación: {len(self.semantic_track_ids_filtered)}")
            else:
                semantic_df = pd.DataFrame({
                    'track_id': self.semantic_track_ids_filtered,
                    'embedding_index': range(len(self.semantic_track_ids_filtered))
                })
            
            if musical_duplicates > 0:
                print("Eliminando duplicados musicales...")
                # Mantener primer occurrence de cada track_id
                self.musical_dataset_filtered = self.musical_dataset_filtered.drop_duplicates(
                    subset=['track_id'], keep='first'
                ).copy()
                print(f"Registros musicales únicos: {len(self.musical_dataset_filtered)}")
            
            # Merge para alinear datos (ahora sin duplicados)
            aligned_data = self.musical_dataset_filtered.merge(
                semantic_df, 
                on='track_id', 
                how='inner'
            ).copy()
            
            print(f"Datos alineados (post-deduplicación): {len(aligned_data)} canciones")
            
            # Verificar unicidad final
            unique_track_ids = aligned_data['track_id'].nunique()
            total_records = len(aligned_data)
            
            if unique_track_ids != total_records:
                raise ValueError(f"Aún hay duplicados: {unique_track_ids} únicos vs {total_records} registros")
            
            # Extraer embeddings alineados
            aligned_embedding_indices = aligned_data['embedding_index'].values
            aligned_embeddings = self.semantic_embeddings_filtered[aligned_embedding_indices]
            
            # Extraer características musicales alineadas
            aligned_musical_features = aligned_data[self.available_musical_features].values
            
            # Extraer metadatos
            aligned_metadata = aligned_data[[
                'track_id', 'track_name', 'track_artist', 'lyrics',
                'track_popularity', 'playlist_genre', 'playlist_subgenre'
            ]].copy()
            
            self.aligned_embeddings = aligned_embeddings
            self.aligned_musical_features = aligned_musical_features
            self.aligned_metadata = aligned_metadata
            self.aligned_track_ids = aligned_data['track_id'].values
            
            print(f"Alineación completada:")
            print(f"  Embeddings: {self.aligned_embeddings.shape}")
            print(f"  Características musicales: {self.aligned_musical_features.shape}")
            print(f"  Metadatos: {len(self.aligned_metadata)}")
            print(f"  Track IDs únicos: {len(set(self.aligned_track_ids))}")
            
            return True
            
        except Exception as e:
            print(f"Error alineando datasets: {e}")
            return False
    
    def normalize_musical_features(self):
        """Normalizar características musicales usando StandardScaler."""
        try:
            print("Normalizando características musicales...")
            
            # Verificar datos no-nulos
            has_nulls = pd.isnull(self.aligned_musical_features).any()
            if has_nulls:
                print("Advertencia: Detectados valores nulos en características musicales")
                # Rellenar nulos con mediana
                musical_features_df = pd.DataFrame(self.aligned_musical_features, 
                                                 columns=self.available_musical_features)
                musical_features_df = musical_features_df.fillna(musical_features_df.median())
                self.aligned_musical_features = musical_features_df.values
            
            # Aplicar StandardScaler
            self.musical_scaler = StandardScaler()
            self.aligned_musical_features_normalized = self.musical_scaler.fit_transform(
                self.aligned_musical_features
            )
            
            print(f"Características musicales normalizadas: {self.aligned_musical_features_normalized.shape}")
            
            # Verificar normalización
            means = np.mean(self.aligned_musical_features_normalized, axis=0)
            stds = np.std(self.aligned_musical_features_normalized, axis=0)
            
            print(f"Verificación normalización:")
            print(f"  Media promedio: {np.mean(means):.6f} (esperado: ~0)")
            print(f"  Std promedio: {np.mean(stds):.6f} (esperado: ~1)")
            
            return True
            
        except Exception as e:
            print(f"Error normalizando características musicales: {e}")
            return False
    
    def validate_dataset_integrity(self):
        """Validar integridad del dataset unificado."""
        try:
            print("Validando integridad del dataset...")
            
            # Verificar consistencia de tamaños
            n_samples = len(self.aligned_track_ids)
            checks = {
                'embeddings_shape': self.aligned_embeddings.shape[0] == n_samples,
                'musical_raw_shape': self.aligned_musical_features.shape[0] == n_samples,
                'musical_norm_shape': self.aligned_musical_features_normalized.shape[0] == n_samples,
                'metadata_shape': len(self.aligned_metadata) == n_samples,
                'embedding_dimensions': self.aligned_embeddings.shape[1] == 384,
                'musical_dimensions': self.aligned_musical_features.shape[1] == len(self.available_musical_features)
            }
            
            all_passed = all(checks.values())
            
            print(f"Validación de integridad:")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name}: {passed}")
            
            if not all_passed:
                raise ValueError("Falló validación de integridad")
            
            # Verificar unicidad de track_ids
            unique_tracks = len(set(self.aligned_track_ids))
            if unique_tracks != n_samples:
                raise ValueError(f"Track IDs duplicados: {unique_tracks} únicos vs {n_samples} total")
            
            print(f"✅ Integridad validada: {n_samples} canciones únicas")
            
            return True
            
        except Exception as e:
            print(f"Error validando integridad: {e}")
            return False
    
    def build_unified_dataset(self):
        """Construir estructura final del dataset unificado."""
        try:
            print("Construyendo dataset unificado...")
            
            # Compilar dataset unificado
            self.unified_dataset = {
                'metadata': {
                    'creation_timestamp': self.timestamp,
                    'builder_version': '1.0',
                    'source_semantic_embeddings': str(self.semantic_embeddings_path),
                    'source_musical_dataset': str(self.musical_dataset_path),
                    'valid_track_ids_source': str(self.valid_track_ids_path),
                    'sample_size': len(self.aligned_track_ids),
                    'semantic_dimensions': self.aligned_embeddings.shape[1],
                    'musical_dimensions': len(self.available_musical_features),
                    'musical_features_used': self.available_musical_features
                },
                
                'data': {
                    'track_ids': self.aligned_track_ids,
                    'semantic_embeddings': self.aligned_embeddings,  # 384D BERT
                    'musical_features_raw': self.aligned_musical_features,  # 12D raw
                    'musical_features_normalized': self.aligned_musical_features_normalized,  # 12D normalized
                    'track_metadata': self.aligned_metadata
                },
                
                'preprocessing': {
                    'musical_scaler': self.musical_scaler,
                    'musical_feature_names': self.available_musical_features,
                    'normalization_method': 'StandardScaler',
                    'semantic_preprocessing': 'BERT paraphrase-multilingual-MiniLM-L12-v2'
                },
                
                'statistics': {
                    'semantic_embeddings_stats': {
                        'mean': float(np.mean(self.aligned_embeddings)),
                        'std': float(np.std(self.aligned_embeddings)),
                        'min': float(np.min(self.aligned_embeddings)),
                        'max': float(np.max(self.aligned_embeddings))
                    },
                    'musical_features_raw_stats': {
                        'means': self.aligned_musical_features.mean(axis=0).tolist(),
                        'stds': self.aligned_musical_features.std(axis=0).tolist(),
                        'mins': self.aligned_musical_features.min(axis=0).tolist(),
                        'maxs': self.aligned_musical_features.max(axis=0).tolist()
                    },
                    'musical_features_normalized_stats': {
                        'means': self.aligned_musical_features_normalized.mean(axis=0).tolist(),
                        'stds': self.aligned_musical_features_normalized.std(axis=0).tolist()
                    },
                    'genre_distribution': self.aligned_metadata['playlist_genre'].value_counts().to_dict(),
                    'dataset_size_mb': self._calculate_dataset_size()
                }
            }
            
            print(f"Dataset unificado construido:")
            print(f"  Muestra: {self.unified_dataset['metadata']['sample_size']:,} canciones")
            print(f"  Dimensiones semánticas: {self.unified_dataset['metadata']['semantic_dimensions']}")
            print(f"  Dimensiones musicales: {self.unified_dataset['metadata']['musical_dimensions']}")
            print(f"  Tamaño estimado: {self.unified_dataset['statistics']['dataset_size_mb']:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error construyendo dataset unificado: {e}")
            return False
    
    def _calculate_dataset_size(self):
        """Calcular tamaño estimado del dataset en MB."""
        size_bytes = 0
        
        # Embeddings semánticos
        size_bytes += self.aligned_embeddings.nbytes
        
        # Características musicales
        size_bytes += self.aligned_musical_features.nbytes
        size_bytes += self.aligned_musical_features_normalized.nbytes
        
        # Track IDs (estimación)
        size_bytes += len(self.aligned_track_ids) * 50  # ~50 bytes por track_id
        
        return size_bytes / (1024 * 1024)
    
    def save_unified_dataset(self):
        """Guardar dataset unificado en archivos."""
        try:
            print("Guardando dataset unificado...")
            
            output_dir = Path(__file__).parent
            
            # Guardar dataset completo en pickle
            dataset_path = output_dir / f"unified_multimodal_dataset_{self.timestamp}.pkl"
            with open(dataset_path, 'wb') as f:
                pickle.dump(self.unified_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Dataset unificado guardado en: {dataset_path}")
            
            # Guardar metadatos en JSON (para inspección rápida)
            metadata_path = output_dir / f"unified_dataset_metadata_{self.timestamp}.json"
            metadata_for_json = {
                'metadata': self.unified_dataset['metadata'],
                'statistics': self.unified_dataset['statistics']
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_for_json, f, indent=2, ensure_ascii=False)
            
            print(f"Metadatos guardados en: {metadata_path}")
            
            # Guardar arrays principales por separado (para acceso eficiente)
            arrays_dir = output_dir / f"arrays_{self.timestamp}"
            arrays_dir.mkdir(exist_ok=True)
            
            np.save(arrays_dir / "track_ids.npy", self.aligned_track_ids)
            np.save(arrays_dir / "semantic_embeddings.npy", self.aligned_embeddings)
            np.save(arrays_dir / "musical_features_raw.npy", self.aligned_musical_features)
            np.save(arrays_dir / "musical_features_normalized.npy", self.aligned_musical_features_normalized)
            
            print(f"Arrays guardados en: {arrays_dir}")
            
            # Crear script de carga rápida
            loader_script = f"""#!/usr/bin/env python3
# Script de carga rápida para dataset unificado multimodal
# Generado automáticamente: {self.timestamp}

import numpy as np
import pickle
from pathlib import Path

def load_unified_dataset():
    \"\"\"Cargar dataset unificado multimodal completo.\"\"\"
    base_path = Path(__file__).parent
    dataset_path = base_path / "unified_multimodal_dataset_{self.timestamp}.pkl"
    
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)

def load_arrays_only():
    \"\"\"Cargar solo arrays principales para análisis rápido.\"\"\"
    base_path = Path(__file__).parent / "arrays_{self.timestamp}"
    
    return {{
        'track_ids': np.load(base_path / "track_ids.npy"),
        'semantic_embeddings': np.load(base_path / "semantic_embeddings.npy"),
        'musical_features_raw': np.load(base_path / "musical_features_raw.npy"),
        'musical_features_normalized': np.load(base_path / "musical_features_normalized.npy")
    }}

if __name__ == "__main__":
    print("Cargando dataset unificado...")
    dataset = load_unified_dataset()
    print(f"Dataset cargado: {{dataset['metadata']['sample_size']:,}} canciones")
    print(f"Dimensiones: {{dataset['metadata']['semantic_dimensions']}}D semántico, {{dataset['metadata']['musical_dimensions']}}D musical")
"""
            
            loader_path = output_dir / f"load_unified_dataset_{self.timestamp}.py"
            with open(loader_path, 'w', encoding='utf-8') as f:
                f.write(loader_script)
            
            print(f"Script de carga generado: {loader_path}")
            
            return True
            
        except Exception as e:
            print(f"Error guardando dataset: {e}")
            return False
    
    def print_summary(self):
        """Imprimir resumen del dataset unificado."""
        print("\n" + "="*80)
        print("RESUMEN DEL DATASET MULTIMODAL UNIFICADO")
        print("="*80)
        
        metadata = self.unified_dataset['metadata']
        stats = self.unified_dataset['statistics']
        
        print(f"\nCONFIGURACIÓN DEL DATASET:")
        print(f"  Muestra total: {metadata['sample_size']:,} canciones")
        print(f"  Dimensiones semánticas: {metadata['semantic_dimensions']}D (BERT)")
        print(f"  Dimensiones musicales: {metadata['musical_dimensions']}D (Spotify)")
        print(f"  Tamaño estimado: {stats['dataset_size_mb']:.1f} MB")
        
        print(f"\nCARACTERÍSTICAS MUSICALES:")
        for i, feature in enumerate(metadata['musical_features_used']):
            raw_mean = stats['musical_features_raw_stats']['means'][i]
            norm_mean = stats['musical_features_normalized_stats']['means'][i]
            print(f"  {feature}: raw_mean={raw_mean:.3f}, norm_mean={norm_mean:.6f}")
        
        print(f"\nDISTRIBUCIÓN DE GÉNEROS:")
        for genre, count in stats['genre_distribution'].items():
            percentage = (count / metadata['sample_size']) * 100
            print(f"  {genre}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nESTADÍSTICAS SEMÁNTICAS:")
        sem_stats = stats['semantic_embeddings_stats']
        print(f"  Media: {sem_stats['mean']:.6f}")
        print(f"  Desviación estándar: {sem_stats['std']:.6f}")
        print(f"  Rango: [{sem_stats['min']:.6f}, {sem_stats['max']:.6f}]")
        
        print(f"\n✅ Dataset multimodal unificado completado exitosamente")
        print(f"🎯 Listo para evaluación de clustering readiness")
        print("="*80)
    
    def run_complete_build(self):
        """Ejecutar construcción completa del dataset unificado."""
        print("Iniciando construcción de dataset multimodal unificado...")
        
        # Secuencia de construcción
        steps = [
            ("Cargar track IDs multimodales", self.load_multimodal_track_ids),
            ("Filtrar datos semánticos", self.load_and_filter_semantic_data),
            ("Filtrar datos musicales", self.load_and_filter_musical_data),
            ("Alinear datasets", self.align_datasets),
            ("Normalizar características", self.normalize_musical_features),
            ("Validar integridad", self.validate_dataset_integrity),
            ("Construir dataset", self.build_unified_dataset),
            ("Guardar dataset", self.save_unified_dataset)
        ]
        
        for step_name, step_function in steps:
            print(f"\n[PASO] {step_name}...")
            success = step_function()
            if not success:
                print(f"❌ Error en paso: {step_name}")
                return False
            print(f"✅ {step_name} completado")
        
        # Mostrar resumen final
        self.print_summary()
        
        return True


def main():
    """Función principal de ejecución."""
    print("CONSTRUCCIÓN DE DATASET MULTIMODAL UNIFICADO")
    print("="*60)
    
    # Inicializar constructor
    builder = UnifiedMultimodalDatasetBuilder()
    
    # Ejecutar construcción completa
    success = builder.run_complete_build()
    
    if success:
        print(f"\n🎉 Dataset unificado creado exitosamente")
        print(f"📊 Archivos disponibles en: phase1_dataset_unification/")
        return builder
    else:
        print(f"\n💥 Construcción del dataset falló")
        return None


if __name__ == "__main__":
    builder = main()