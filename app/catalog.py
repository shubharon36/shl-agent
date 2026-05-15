"""
Catalog loading, indexing, and semantic search over the SHL assessment catalog.
Uses sentence-transformers for embeddings and FAISS for vector search.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional


# Mapping from catalog 'keys' to short test type codes
KEY_TO_TEST_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Simulations": "S",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


class AssessmentCatalog:
    """Manages the SHL assessment catalog with semantic search capabilities."""

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            # Look for dataset.json relative to this file's directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "dataset.json")

        self.assessments: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embedder: Optional[SentenceTransformer] = None
        self._loaded = False

        self._load_catalog(dataset_path)
        self._build_index()

    def _load_catalog(self, path: str):
        """Load assessments from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Only keep assessments with status "ok"
        self.assessments = [a for a in raw if a.get("status") == "ok"]

        # Pre-compute test_type for each assessment
        for a in self.assessments:
            a["test_type"] = self._compute_test_type(a.get("keys", []))
            a["search_text"] = self._build_search_text(a)

        print(f"Loaded {len(self.assessments)} assessments from catalog.")

    def _compute_test_type(self, keys: List[str]) -> str:
        """Convert catalog keys list to comma-separated test type codes."""
        codes = []
        for key in keys:
            code = KEY_TO_TEST_TYPE.get(key, "")
            if code and code not in codes:
                codes.append(code)
        return ",".join(codes) if codes else "K"

    def _build_search_text(self, assessment: Dict[str, Any]) -> str:
        """Build a rich text representation for embedding."""
        parts = [
            f"Assessment: {assessment.get('name', '')}",
            f"Description: {assessment.get('description', '')}",
            f"Categories: {', '.join(assessment.get('keys', []))}",
            f"Job Levels: {', '.join(assessment.get('job_levels', []))}",
            f"Duration: {assessment.get('duration', 'N/A')}",
            f"Remote: {assessment.get('remote', 'N/A')}",
            f"Adaptive: {assessment.get('adaptive', 'N/A')}",
        ]
        if assessment.get("languages"):
            parts.append(f"Languages: {', '.join(assessment['languages'][:5])}")
        return " | ".join(parts)

    def _build_index(self):
        """Build FAISS index from assessment search texts."""
        print("Building FAISS index...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [a["search_text"] for a in self.assessments]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Use inner-product (cosine similarity with normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self._loaded = True
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim}).")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Semantic search over the catalog. Returns top_k assessments."""
        if not self._loaded:
            return []

        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.assessments):
                continue
            assessment = self.assessments[idx].copy()
            assessment["relevance_score"] = float(scores[0][i])
            results.append(assessment)

        return results

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up an assessment by exact name (case-insensitive)."""
        name_lower = name.lower().strip()
        for a in self.assessments:
            if a["name"].lower().strip() == name_lower:
                return a
        return None

    def get_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Look up multiple assessments by name (case-insensitive, fuzzy)."""
        results = []
        for name in names:
            result = self.get_by_name(name)
            if result:
                results.append(result)
            else:
                # Try partial match
                name_lower = name.lower().strip()
                for a in self.assessments:
                    if name_lower in a["name"].lower() or a["name"].lower() in name_lower:
                        results.append(a)
                        break
        return results

    def format_for_context(self, assessments: List[Dict[str, Any]]) -> str:
        """Format a list of assessments as context text for the LLM."""
        if not assessments:
            return "No assessments found."

        lines = []
        for i, a in enumerate(assessments, 1):
            langs = ", ".join(a.get("languages", [])[:3])
            if len(a.get("languages", [])) > 3:
                langs += f" (+{len(a['languages']) - 3} more)"

            lines.append(
                f"{i}. {a['name']}\n"
                f"   URL: {a['link']}\n"
                f"   Test Type: {a['test_type']}\n"
                f"   Categories: {', '.join(a.get('keys', []))}\n"
                f"   Job Levels: {', '.join(a.get('job_levels', []))}\n"
                f"   Duration: {a.get('duration', 'N/A')}\n"
                f"   Remote: {a.get('remote', 'N/A')}\n"
                f"   Adaptive: {a.get('adaptive', 'N/A')}\n"
                f"   Languages: {langs}\n"
                f"   Description: {a.get('description', 'N/A')}\n"
            )
        return "\n".join(lines)

    def get_all_formatted_compact(self) -> str:
        """Return a compact representation of ALL assessments for full catalog awareness."""
        lines = []
        for a in self.assessments:
            lines.append(f"- {a['name']} | {a['test_type']} | {', '.join(a.get('keys',[]))} | {', '.join(a.get('job_levels',[]))} | {a.get('duration','N/A')} | {a['link']}")
        return "\n".join(lines)
