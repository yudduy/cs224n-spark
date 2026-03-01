import numpy as np
import json
from pathlib import Path

BASE = Path("data/modal_runs/gen_traces_full")

# Load embeddings for all 4 models
models = ["r1-distill", "nemotron-v1", "nemotron-v2", "nemotron-brorl"]
labels = ["R1-Distill", "v1 (2K)", "v2 (3K)", "BroRL"]

with open(BASE / "problems.json") as f:
    problems = json.load(f)

all_embs = {}
all_traces = {}
for m in models:
    all_embs[m] = np.load(BASE / m / "embeddings.npy")
    with open(BASE / m / "traces.json") as f:
        data = json.load(f)
        all_traces[m] = data["problems"]  # list of {problem_id, rollouts: [...]}

K = 64
n_problems = 60

print("=" * 70)
print("DIAGNOSTIC 1: Embedding spread (intra-model cosine similarity)")
print("=" * 70)
print("Are RL-trained model embeddings more self-similar?")
print()

for tier_name, tier_range in [("easy", range(0,20)), ("medium", range(20,40)), ("hard", range(40,60))]:
    print(f"  {tier_name.upper()}:")
    for mi, m in enumerate(models):
        sims = []
        for p in tier_range:
            emb = all_embs[m][p*K:(p+1)*K]
            # Normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb_n = emb / (norms + 1e-10)
            # Mean pairwise cosine similarity (excluding diagonal)
            sim_matrix = emb_n @ emb_n.T
            n = sim_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            mean_sim = sim_matrix[mask].mean()
            sims.append(mean_sim)
        print(f"    {labels[mi]:15s}: mean_cos_sim = {np.mean(sims):.4f} (std={np.std(sims):.4f})")
    print()

print("=" * 70)
print("DIAGNOSTIC 2: Cross-model embedding distance")
print("=" * 70)
print("Do different models embed the same problem differently?")
print()

for p_idx in [0, 15, 23, 30, 45]:  # Sample problems
    prob = problems[p_idx]
    print(f"  Problem {p_idx} [{prob['tier']}/{prob['subject']}]:")
    # Get centroid for each model
    centroids = {}
    for m in models:
        emb = all_embs[m][p_idx*K:(p_idx+1)*K]
        centroids[m] = emb.mean(axis=0)
    # Pairwise cosine between model centroids
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            c1 = centroids[models[i]]
            c2 = centroids[models[j]]
            cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            print(f"    {labels[i]:12s} vs {labels[j]:12s}: cos = {cos:.4f}")
    print()

print("=" * 70)
print("DIAGNOSTIC 3: Answer diversity vs embedding diversity")
print("=" * 70)
print("Do traces with different answers actually have different embeddings?")
print()

for p_idx in [0, 15, 23, 30, 45]:
    prob = problems[p_idx]
    print(f"  Problem {p_idx} [{prob['tier']}/{prob['subject']}]:")
    for mi, m in enumerate(models):
        prob_data = all_traces[m][p_idx]
        rollouts = prob_data["rollouts"]
        emb = all_embs[m][p_idx*K:(p_idx+1)*K]
        # Get answers
        answers = [r["final_answer"] for r in rollouts]

        unique_ans = set(answers)
        if len(unique_ans) <= 1:
            print(f"    {labels[mi]:15s}: 1 unique answer -> no answer-based separation test")
            continue

        # Group embeddings by answer
        ans_groups = {}
        for k_i in range(K):
            a = answers[k_i]
            if a not in ans_groups:
                ans_groups[a] = []
            ans_groups[a].append(emb[k_i])

        # Within-answer similarity vs between-answer similarity
        within_sims = []
        between_sims = []
        ans_list = list(ans_groups.keys())
        for a in ans_list:
            vecs = np.array(ans_groups[a])
            if len(vecs) > 1:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs_n = vecs / (norms + 1e-10)
                sim = vecs_n @ vecs_n.T
                mask_inner = ~np.eye(len(vecs), dtype=bool)
                within_sims.extend(sim[mask_inner].tolist())

        for i in range(len(ans_list)):
            for j in range(i+1, len(ans_list)):
                v1 = np.array(ans_groups[ans_list[i]])
                v2 = np.array(ans_groups[ans_list[j]])
                n1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-10)
                n2 = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-10)
                cross = n1 @ n2.T
                between_sims.extend(cross.flatten().tolist())

        w = np.mean(within_sims) if within_sims else float('nan')
        b = np.mean(between_sims) if between_sims else float('nan')
        print(f"    {labels[mi]:15s}: {len(unique_ans)} answers, within_sim={w:.4f}, between_sim={b:.4f}, gap={w-b:.4f}")
    print()

print("=" * 70)
print("DIAGNOSTIC 4: Trace length variance")
print("=" * 70)
print("Are RL models generating more uniform-length traces?")
print()

for mi, m in enumerate(models):
    lengths = []
    for prob_data in all_traces[m]:
        for r in prob_data["rollouts"]:
            lengths.append(len(r["response"]))
    print(f"  {labels[mi]:15s}: mean_len={np.mean(lengths):.0f}, std={np.std(lengths):.0f}, cv={np.std(lengths)/np.mean(lengths):.3f}")

print()
print("=" * 70)
print("DIAGNOSTIC 5: Random baseline -- how many clusters does noise get?")
print("=" * 70)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Generate random 384-dim vectors and cluster them same way
np.random.seed(42)
random_emb = np.random.randn(256, 384).astype(np.float32)
random_emb = random_emb / (np.linalg.norm(random_emb, axis=1, keepdims=True) + 1e-10)
dist = 1 - (random_emb @ random_emb.T)
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, 2)

best_k, best_sil = 2, -1
for k in range(2, 8):
    clust = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labs = clust.fit_predict(dist)
    sil = silhouette_score(dist, labs, metric="precomputed")
    if sil > best_sil:
        best_k, best_sil = k, sil
    print(f"  k={k}: silhouette={sil:.4f}")

print(f"  Best k for random noise: {best_k} (sil={best_sil:.4f})")

print()
print("=" * 70)
print("DIAGNOSTIC 6: Embedding variance by principal components")
print("=" * 70)
print("How much variance do MiniLM embeddings capture per problem?")
print()

from sklearn.decomposition import PCA

for mi, m in enumerate(models):
    var_explained_1 = []
    var_explained_5 = []
    for p in range(n_problems):
        emb = all_embs[m][p*K:(p+1)*K]
        pca = PCA(n_components=min(10, K))
        pca.fit(emb)
        var_explained_1.append(pca.explained_variance_ratio_[0])
        var_explained_5.append(sum(pca.explained_variance_ratio_[:5]))
    print(f"  {labels[mi]:15s}: PC1 explains {np.mean(var_explained_1)*100:.1f}% (std={np.std(var_explained_1)*100:.1f}%), "
          f"PC1-5 explain {np.mean(var_explained_5)*100:.1f}%")

print()
print("=" * 70)
print("DIAGNOSTIC 7: Real data clustering with silhouette (replicating Phase 2)")
print("=" * 70)
print("What does silhouette-optimal k look like across models?")
print()

for mi, m in enumerate(models):
    ks = []
    sils = []
    for p in range(n_problems):
        emb = all_embs[m][p*K:(p+1)*K]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_n = emb / (norms + 1e-10)
        dist_mat = 1 - (emb_n @ emb_n.T)
        np.fill_diagonal(dist_mat, 0)
        dist_mat = np.clip(dist_mat, 0, 2)

        best_k_p, best_sil_p = 1, -1  # default: 1 cluster
        for k in range(2, 8):
            clust = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labs = clust.fit_predict(dist_mat)
            sil = silhouette_score(dist_mat, labs, metric="precomputed")
            if sil > best_sil_p:
                best_k_p, best_sil_p = k, sil
        ks.append(best_k_p)
        sils.append(best_sil_p)

    k_counts = {}
    for kk in ks:
        k_counts[kk] = k_counts.get(kk, 0) + 1
    print(f"  {labels[mi]:15s}: mean_k={np.mean(ks):.2f}, k_distribution={dict(sorted(k_counts.items()))}, mean_sil={np.mean(sils):.4f}")

print()
print("=" * 70)
print("DIAGNOSTIC 8: Embedding collapse check (norm and spread)")
print("=" * 70)

for mi, m in enumerate(models):
    all_norms = []
    mean_dists = []
    for p in range(n_problems):
        emb = all_embs[m][p*K:(p+1)*K]
        norms = np.linalg.norm(emb, axis=1)
        all_norms.extend(norms.tolist())
        # Mean pairwise L2 distance
        centroid = emb.mean(axis=0)
        dists = np.linalg.norm(emb - centroid, axis=1)
        mean_dists.append(np.mean(dists))
    print(f"  {labels[mi]:15s}: mean_norm={np.mean(all_norms):.4f} (std={np.std(all_norms):.4f}), "
          f"mean_dist_to_centroid={np.mean(mean_dists):.4f} (std={np.std(mean_dists):.4f})")
