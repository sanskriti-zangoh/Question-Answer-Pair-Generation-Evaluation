import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from langchain_ollama import OllamaEmbeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

def get_latent_space(dir: str = "result/test5", filename: str = 'qna_context_embeddings.csv', plot_filename: str = 'latent_space_plot.png'):
    # Load the data from the JSON file
    embedded_queries = pd.read_csv(os.path.join(dir, "qna_context_embeddings.csv"))
    question_embeddings = embedded_queries['question_embeddings'].to_list()
    # PCA on embeddings to reduce to 10-dim
    pca = PCA(n_components=10)
    question_embeddings_reduced = pca.fit_transform(question_embeddings)

    # Determine an appropriate value for perplexity
    n_samples = len(question_embeddings_reduced)
    perplexity = min(30, n_samples - 1)

    # TSNE on embeddings to reduce to 2-dim
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=2024)
    lower_dim_embeddings = tsne.fit_transform(question_embeddings_reduced)
    labels = np.concatenate(
        [
            np.full(len(lower_dim_embeddings), "generated"),

        ]
    )
    data = pd.DataFrame(
        {"x": lower_dim_embeddings[:, 0], "y": lower_dim_embeddings[:, 1], "label": labels}
    )
    sns.scatterplot(data=data, x="x", y="y", hue="label")

    # Save the plot
    plt.savefig(os.path.join(dir, plot_filename))
    plt.close()