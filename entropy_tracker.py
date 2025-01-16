from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean


FOLDER = "data/02-processed/activation_tracker.pkl"
SAVE_FOLDER = "plots/entropy"

def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def calculate_lower_diagonal_similarity_distance(embeddings, states):
    embeddings = embeddings[0]
    # Matriz de similitud para embeddings
    embedding_similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            embedding_similarity_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])

    # Matriz de distancia L2 para states_15
    states_distance_matrix = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            states_distance_matrix[i, j] = euclidean(states[i], states[j])

    embedding_similarity_matrix = normalize(embedding_similarity_matrix)
    states_distance_matrix = normalize(states_distance_matrix)

    # Extraer los valores de la diagonal inferior
    results = []
    for i in range(1, len(embeddings)):
        for j in range(i):
            results.append(((i, j), embedding_similarity_matrix[i, j], states_distance_matrix[i, j]))

    return results

def process_dataframe(df):
    df["token_pair_similarity_distance"] = df.apply(
        lambda row: calculate_lower_diagonal_similarity_distance(row["embeddings"], row["states_15"]), axis=1
    )
    return df

# Visualización de resultados
def plot_similarity_distance(df, title="Cosine Similarity vs L2 Distance"):
    all_points = []
    for row in df["token_pair_similarity_distance"]:
        for _, similarity, distance in row:
            all_points.append((similarity, distance))

    # Convertimos a un array para facilitar el manejo
    all_points = np.array(all_points)
    similarities = all_points[:, 0]
    distances = all_points[:, 1]

    # Gráfico 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(similarities, distances, alpha=0.7)
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1)  # Línea horizontal en la mitad
    plt.axvline(0.5, color='black', linestyle='--', linewidth=1)  # Línea vertical en la mitad

    # Colorear los cuadrantes
    plt.fill_betweenx([0, 0.5], 0, 0.5, color=(1.0, 0.9, 0.9), alpha=0.5, label = "High entropy")  # Cuadrante 1,1 verde claro
    plt.fill_betweenx([0.5, 1], 0.5, 1, color=(1.0, 0.9, 0.9), alpha=0.5)  # Cuadrante 2,2 verde claro
    plt.fill_betweenx([0, 0.5], 0.5, 1, color=(0.9, 1.0, 0.9), alpha=0.5, label = "Low entropy")  # Cuadrante 1,2 rojo claro
    plt.fill_betweenx([0.5, 1], 0, 0.5, color=(0.9, 1.0, 0.9), alpha=0.5)  # Cuadrante 2,1 rojo claro

    # Gráfico 2D
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("Cosine Similarity Embeddings")
    plt.ylabel("L2 Distance Last Hidden State")
    plt.grid(True)
    # SAVE
    plt.savefig(f"{SAVE_FOLDER}/{title}.png")

if __name__ == "__main__":
    # Procesamos el DataFrame
    dataset = pd.read_pickle(FOLDER)
    processed_df = process_dataframe(dataset)

    # Graficamos los resultados
    plot_similarity_distance(processed_df, title="Cosine Similarity vs L2 Distance Spanish and English")

    # filtramos por idioma
    spanish_df = dataset[dataset["language"] == "Spanish"]
    english_df = dataset[dataset["language"] == "English"]

    # Procesamos los DataFrames
    spanish_df = process_dataframe(spanish_df)
    english_df = process_dataframe(english_df)
    # Graficamos los resultados para el idioma español
    
    plot_similarity_distance(spanish_df, title="Cosine Similarity vs L2 Distance Spanish")

    # Graficamos los resultados para el idioma inglés
    plot_similarity_distance(english_df, title="Cosine Similarity vs L2 Distance English")

