# openl3_test.py
import openl3
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
import os


def analyze_with_openl3(audio_file):
    """
    Analizar audio con OpenL3 y generar embeddings
    """
    print(f"🎵 Analizando: {audio_file}")
    
    try:
        # Cargar audio
        audio, sr = librosa.load(audio_file, sr=None)
        print(f"✅ Audio cargado: {len(audio)/sr:.1f} segundos, {sr} Hz")
        
        # Generar embeddings con OpenL3
        print("🤖 Generando embeddings con OpenL3...")
        import audio_analysis.aa_openl3 as aa_openl3
        import tempfile
        
        # Guardar audio temporal
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio, sr)
        
        # Generar embeddings con OpenL3
        print("🤖 Generando embeddings con OpenL3...")


        # Procesar con OpenL3
        openl3.process_audio_file(
            temp_file,
            output_dir=".",
            verbose=True
        )

        # Limpiar archivo temporal
        os.remove(temp_file)

        # Cargar el embedding generado
        embedding = np.load("temp_audio.npz")['embedding']
        
        
        print(f"✅ Embedding generado:")
        print(f"   📊 Forma: {embedding.shape}")
        print(f"   🎯 Dimensiones: {embedding.shape[1]}")
        print(f"   ⏱️  Segmentos: {embedding.shape[0]}")
        
        # Promedio temporal - una representación por canción
        song_embedding = np.mean(embedding, axis=0)
        
        print(f"\n📊 ESTADÍSTICAS DEL EMBEDDING:")
        print(f"   🎵 Vector final: {song_embedding.shape}")
        print(f"   📈 Rango: [{song_embedding.min():.3f}, {song_embedding.max():.3f}]")
        print(f"   📊 Media: {song_embedding.mean():.3f}")
        print(f"   📏 Norma L2: {np.linalg.norm(song_embedding):.3f}")
        
        # Guardar
        np.save('we_will_rock_you_openl3.npy', song_embedding)
        print(f"💾 Embedding guardado: we_will_rock_you_openl3.npy")
        
        return song_embedding
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def compare_embeddings(embedding1, embedding2):
    """
    Comparar dos embeddings de audio
    """
    if embedding1 is None or embedding2 is None:
        return None
    
    # Calcular similitud coseno
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
    # Distancia euclidiana
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    print(f"🎯 SIMILITUD ENTRE AUDIOS:")
    print(f"   Similitud coseno: {similarity:.6f}")
    print(f"   Distancia euclidiana: {euclidean_dist:.6f}")
    
    return similarity

def main():
    # Archivo de audio principal
    audio_file = "we_will_rock_you.mp3"  # 🔧 CAMBIAR POR TU RUTA
    
    print("🎵 ANÁLISIS DE AUDIO CON OPENL3")
    print("=" * 50)
    
    # Analizar We Will Rock You
    queen_embedding = analyze_with_openl3(audio_file)
    
    if queen_embedding is not None:
        print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
        print(f"   🎵 We Will Rock You analizada exitosamente")
        print(f"   🤖 Embedding de 512 dimensiones generado")
        print(f"   📊 Listo para comparar con otras canciones")
        
        # Si tienes otra canción para comparar:
        # other_file = "otra_cancion.mp3"
        # other_embedding = analyze_with_openl3(other_file)
        # compare_embeddings(queen_embedding, other_embedding)
    
    else:
        print("❌ No se pudo completar el análisis")
        print("💡 Verifica que tengas el archivo MP3")

if __name__ == "__main__":
    main()