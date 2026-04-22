"""
Self-contained HybridSearchEngine for TrendScout.

Reads from DATA/raw/greenhouse/jobs_master.csv — no dependency on
the trendscout_jobboards sub-repo or any vector store. Provides
keyword search and graph search (built on-the-fly from the CSV).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Resolve project root (CODE/utilities/ -> CODE/ -> root)
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from config import DATA_DIR_RAW, DATA_DIR_PROCESSED  # noqa: E402

_JOBS_CSV = os.path.join(DATA_DIR_RAW, "greenhouse", "jobs_master.csv")
_GRAPH_CSV = os.path.join(DATA_DIR_PROCESSED, "job_graph_edges.csv")

# Common tech keywords used to generate REQUIRES edges from descriptions
_TECH_KEYWORDS = [
    "python", "c++", "java", "javascript", "typescript", "rust", "go", "scala",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis",
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn",
    "kubernetes", "docker", "terraform", "ansible", "aws", "gcp", "azure",
    "linux", "git", "ci/cd", "jenkins", "spark", "kafka", "airflow",
    "react", "node.js", "fastapi", "flask", "django",
    "llm", "nlp", "ml", "deep learning", "computer vision", "reinforcement learning",
    "revit", "autocad", "bim", "primavera", "scada",
]


def _build_edges(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Generate graph edges from jobs_master.csv on-the-fly."""
    edges = []
    for _, row in jobs_df.iterrows():
        company = row.get("company")
        title = row.get("title")
        location = row.get("location")
        description = str(row.get("description", "")).lower()

        if pd.notna(company) and pd.notna(title):
            edges.append({"source": company, "relationship": "HIRING_FOR", "target": title})

        if pd.notna(company) and pd.notna(location):
            edges.append({"source": company, "relationship": "LOCATED_IN", "target": str(location)})

        for kw in _TECH_KEYWORDS:
            if kw in description and pd.notna(title):
                edges.append({"source": title, "relationship": "REQUIRES", "target": kw})

    return pd.DataFrame(edges).drop_duplicates() if edges else pd.DataFrame(columns=["source", "relationship", "target"])


class HybridSearchEngine:
    """
    Hybrid search combining keyword matching (0.6) and graph relationships (0.4).
    """

    def __init__(self):
        self.jobs_df: pd.DataFrame | None = None
        self.edges_df: pd.DataFrame | None = None
        self._load_data()

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        if os.path.exists(_JOBS_CSV):
            self.jobs_df = pd.read_csv(_JOBS_CSV)
        else:
            print(f"[HybridSearch] WARNING: jobs CSV not found at {_JOBS_CSV}")
            return

        # Prefer pre-built edges file; fall back to building on-the-fly
        if os.path.exists(_GRAPH_CSV):
            self.edges_df = pd.read_csv(_GRAPH_CSV)
        else:
            self.edges_df = _build_edges(self.jobs_df)

    # ------------------------------------------------------------------
    def keyword_search(self, query: str, n_results: int = 5) -> List[Dict]:
        if self.jobs_df is None:
            return []

        keywords = query.lower().split()
        scored = []
        for idx, row in self.jobs_df.iterrows():
            text = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
            score = sum(text.count(kw) for kw in keywords)
            if score > 0:
                scored.append((idx, score, row))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (_, score, row) in enumerate(scored[:n_results], 1):
            results.append({
                "rank": rank,
                "method": "keyword",
                "score": score,
                "title": row.get("title", "N/A"),
                "company": row.get("company", "N/A"),
                "url": row.get("job_url", "N/A"),
                "snippet": str(row.get("description", ""))[:200],
                "role_category": row.get("role_category", "N/A"),
                "location": row.get("location", "N/A"),
            })
        return results

    # ------------------------------------------------------------------
    def graph_search(self, entity: str, n_results: int = 5) -> List[Dict]:
        if self.edges_df is None or self.jobs_df is None:
            return []

        entity_lower = entity.lower()
        relevant = self.edges_df[
            self.edges_df["source"].str.lower().str.contains(entity_lower, na=False) |
            self.edges_df["target"].str.lower().str.contains(entity_lower, na=False)
        ]

        results = []
        seen = set()
        for _, edge in relevant.head(n_results * 2).iterrows():
            target = edge["target"]
            matches = self.jobs_df[self.jobs_df["title"].str.lower() == target.lower()]
            if matches.empty:
                continue
            job = matches.iloc[0]
            key = (job.get("title", ""), job.get("company", ""))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "rank": len(results) + 1,
                "method": "graph",
                "score": 1.0,
                "title": job.get("title", "N/A"),
                "company": job.get("company", "N/A"),
                "url": job.get("job_url", "N/A"),
                "snippet": str(job.get("description", ""))[:200],
                "role_category": job.get("role_category", "N/A"),
                "location": job.get("location", "N/A"),
                "relationship": edge["relationship"],
            })
            if len(results) >= n_results:
                break
        return results

    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        weights: Dict[str, float] | None = None,
        n_results: int = 5,
    ) -> Dict:
        if weights is None:
            weights = {"keyword": 0.6, "graph": 0.4}

        raw = {
            "keyword": self.keyword_search(query, n_results),
            "graph": self.graph_search(query, n_results),
        }

        combined: Dict[tuple, Dict] = {}
        for method, weight in weights.items():
            for result in raw.get(method, []):
                key = (result["title"], result["company"])
                if key not in combined:
                    combined[key] = {"combined_score": 0.0, "methods": [], "data": result}
                combined[key]["combined_score"] += weight / (result["rank"] + 1)
                if method not in combined[key]["methods"]:
                    combined[key]["methods"].append(method)

        ranked = sorted(combined.items(), key=lambda x: x[1]["combined_score"], reverse=True)

        final = []
        for rank, ((title, company), item) in enumerate(ranked[:n_results], 1):
            final.append({
                "rank": rank,
                "combined_score": round(item["combined_score"], 3),
                "methods": item["methods"],
                **item["data"],
            })

        return {
            "query": query,
            "results": final,
            "total_results": len(final),
        }
