from gensim.models import FastText
import numpy as np
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import sacrebleu

class Line:
    def __init__(self, id, content, scores=None):
        self.id = id
        self.content = content
        self.scores = scores or {}

class Document:
    def __init__(self, lines=None):
        self.lines = lines or []

    @classmethod
    def from_array(cls, array):
        lines = [Line(i, content) for i, content in enumerate(array)]
        return cls(lines)

    def __getitem__(self, id):
        return next((line for line in self.lines if line.id == id), None)

class LinguisticAnomalyDetector:
    def __init__(self, source_doc, target_doc, num_comparisons=5, normalize_scores=True):
        self.source_doc = source_doc
        self.target_doc = target_doc
        self.num_comparisons = num_comparisons
        self.normalize_scores = normalize_scores

    def _normalize_scores(self, scores):
        mean = np.mean(scores)
        std = np.std(scores)
        return (scores - mean) / (std if std != 0 else 1)

    def _train_fasttext(self, lines):
        return FastText(
            [line.content.split() for line in lines],
            min_count=1,
            vector_size=100,
            workers=4
        )

    def _compute_metrics(self, line, comparison_lines, source_comparison_lines):
        combined_lines = comparison_lines + source_comparison_lines
        embeddings = self._train_fasttext(combined_lines)

        line_embedding = self._get_line_embedding(line, embeddings)
        comparison_embeddings = [self._get_line_embedding(l, embeddings) for l in comparison_lines]
        source_line_embedding = self._get_line_embedding(source_comparison_lines[0], embeddings)

        cosine_scores = [self._cosine_similarity(line_embedding, emb) for emb in comparison_embeddings]
        euclidean_scores = [np.linalg.norm(line_embedding - emb) for emb in comparison_embeddings]
        edit_distances = [levenshtein_distance(line.content, l.content) for l in comparison_lines]

        # Calculate CHRF+ score
        chrf_scores = [sacrebleu.sentence_chrf(line.content, [l.content]).score for l in comparison_lines]

        # Calculate BLEU score
        bleu_scores = [sacrebleu.sentence_bleu(line.content, [l.content]).score for l in comparison_lines]

        if self.normalize_scores:
            cosine_scores = self._normalize_scores(cosine_scores)
            euclidean_scores = self._normalize_scores(euclidean_scores)
            edit_distances = self._normalize_scores(edit_distances)
            chrf_scores = self._normalize_scores(chrf_scores)
            bleu_scores = self._normalize_scores(bleu_scores)

        source_cosine = self._cosine_similarity(line_embedding, source_line_embedding)
        source_euclidean = np.linalg.norm(line_embedding - source_line_embedding)

        return {
            'cosine': np.mean(cosine_scores),
            'euclidean': np.mean(euclidean_scores),
            'edit': np.mean(edit_distances),
            'chrf': np.mean(chrf_scores),
            'bleu': np.mean(bleu_scores),
            'source_cosine': source_cosine,
            'source_euclidean': source_euclidean
        }

    def _get_line_embedding(self, line, embeddings):
        words = line.content.split()
        word_embeddings = [embeddings.wv[word] for word in words if word in embeddings.wv]
        return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(embeddings.vector_size)

    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def detect_anomalies(self):
        for line in self.target_doc.lines:
            comparison_ids = np.random.choice(
                [l.id for l in self.target_doc.lines if l.id != line.id],
                size=min(self.num_comparisons, len(self.target_doc.lines) - 1),
                replace=False
            )
            comparison_lines = [self.target_doc[id] for id in comparison_ids]
            source_line = self.source_doc[line.id]

            if source_line:
                source_comparison_lines = [self.source_doc[id] for id in comparison_ids]
                metrics = self._compute_metrics(line, comparison_lines, source_comparison_lines)
                line.scores = {
                    'target': {
                        k: v for k, v in metrics.items() if k != 'source_cosine' and k != 'source_euclidean'
                    },
                    'source': {'cosine': metrics['source_cosine'], 'euclidean': metrics['source_euclidean']},
                    'anomaly': {k: abs(metrics[k] - metrics['source_' + k]) for k in ['cosine', 'euclidean']},
                    'overall': np.mean([abs(metrics[k] - metrics['source_' + k]) for k in ['cosine', 'euclidean']])
                }

    def plot_scores(self, metric='overall', title=None):
        scores = [line.scores.get(metric, 0) for line in self.target_doc.lines]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(scores)), scores, marker='o')
        plt.xlabel('Line')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(title or f'{metric.capitalize()} Scores for Each Line')
        plt.grid(True)
        plt.show()

    def plot_comparison(self, metric='overall', title=None):
        target_scores = [line.scores['target'].get(metric, 0) for line in self.target_doc.lines]
        source_scores = [line.scores['source'].get(metric, 0) for line in self.target_doc.lines]
        anomaly_scores = [line.scores['anomaly'].get(metric, 0) for line in self.target_doc.lines]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(target_scores)), target_scores, marker='o', label='Target')
        plt.plot(range(len(source_scores)), source_scores, marker='o', label='Source')
        plt.plot(range(len(anomaly_scores)), anomaly_scores, marker='o', label='Anomaly')
        plt.xlabel('Line')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(title or f'Comparison of {metric.capitalize()} Scores')
        plt.grid(True)
        plt.legend()
        plt.show()


kjv_lines = [
    "In the beginning God created the heaven and the earth.",
    "And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters.",
    "And God said, Let there be light: and there was light.",
    "And God saw the light, that it was good: and God divided the light from the darkness.",
    "And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day."
]

louis_segond_lines = [
    "Au commencement, Dieu créa les cieux et la terre.",
    "La terre était informe et vide: il y avait des ténèbres à la surface de l'abîme, et l'esprit de Dieu se mouvait au-dessus des eaux.",
    "Dieu dit: Que la lumière soit! Et la lumière fut.",
    "Dieu vit que la lumière était bonne; et Dieu sépara la lumière d'avec les ténèbres.",
    "Dieu appela la lumière jour, et il appela les ténèbres nuit. Ainsi, il y eut un soir, et il y eut un matin: ce fut le premier jour."
]

kjv_doc = Document.from_array(kjv_lines)
louis_segond_doc = Document.from_array(louis_segond_lines)

# Detect anomalies with normalization
detector = LinguisticAnomalyDetector(kjv_doc, louis_segond_doc, normalize_scores=True)
detector.detect_anomalies()

# Plot scores for all metrics
metrics = ['cosine', 'euclidean', 'edit', 'chrf', 'bleu', 'overall']
num_subplots = len(metrics)
fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))

for i, metric in enumerate(metrics):
    scores = [line.scores['target'].get(metric, 0) for line in detector.target_doc.lines]
    axs[i].plot(range(len(scores)), scores, marker='o')
    axs[i].set_xlabel('Line')
    axs[i].set_ylabel(f'{metric.capitalize()} Score')
    axs[i].set_title(f'{metric.capitalize()} Scores for Each Line')
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# Plot comparisons for all metrics
num_subplots = len(metrics)
fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))

for i, metric in enumerate(metrics):
    target_scores = [line.scores['target'].get(metric, 0) for line in detector.target_doc.lines]
    source_scores = [line.scores['source'].get(metric, 0) for line in detector.target_doc.lines]
    anomaly_scores = [line.scores['anomaly'].get(metric, 0) for line in detector.target_doc.lines]

    axs[i].plot(range(len(target_scores)), target_scores, marker='o', label='Target')
    axs[i].plot(range(len(source_scores)), source_scores, marker='o', label='Source')
    axs[i].plot(range(len(anomaly_scores)), anomaly_scores, marker='o', label='Anomaly')
    axs[i].set_xlabel('Line')
    axs[i].set_ylabel(f'{metric.capitalize()} Score')
    axs[i].set_title(f'Comparison of {metric.capitalize()} Scores')
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.show()
