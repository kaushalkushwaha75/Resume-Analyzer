"""
Semantic matching engine.
Computes multi-layered semantic similarity and skill overlap between resume
and job description at overall, section, and skill levels.
Uses TF-IDF character/word N-grams, BM25-inspired term scoring,
morphological stemming, and BERT (if sentence-transformers is installed).
"""

import re
import numpy as np
from typing import Dict, List, Set, Optional

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        denom = norm_a * norm_b.T
        denom = np.where(denom == 0, 1e-10, denom)
        return dot / denom

# Generic job filler words
GENERIC_JOB_WORDS = {
    "experience", "years", "looking", "responsible", "responsibilities",
    "requirements", "candidate", "role", "position", "seeking", "join",
    "team", "work", "working", "skills", "ability", "plus", "bonus",
    "preferred", "required", "ideal", "opportunity", "including",
    "duties", "task", "tasks", "job", "description", "apply", "applicant"
}

# Standard English stop words
STANDARD_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}

ALL_STOP_WORDS = STANDARD_STOP_WORDS | GENERIC_JOB_WORDS

# Lazy-load optional BERT model
_model = None
_model_attempted = False


def _get_model():
    """Lazy-load the sentence-transformers model if available."""
    global _model, _model_attempted
    if _model is None and not _model_attempted:
        _model_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = None
    return _model


def simple_stem(word: str) -> str:
    """
    Lightweight rule-based stemmer for tech/job vocabulary.
    Normalizes 'developer'/'development' -> 'develop', 'engineer'/'engineering' -> 'engineer', etc.
    """
    w = word.lower().strip()
    if len(w) <= 3:
        return w

    irreg = {
        "development": "develop",
        "developer": "develop",
        "developers": "develop",
        "developing": "develop",
        "developed": "develop",
        "engineering": "engineer",
        "engineer": "engineer",
        "engineers": "engineer",
        "engineered": "engineer",
        "management": "manage",
        "manager": "manage",
        "managers": "manage",
        "managing": "manage",
        "analysis": "analy",
        "analyst": "analy",
        "analytics": "analy",
        "analyzing": "analy",
        "analyzed": "analy",
        "programmer": "program",
        "programming": "program",
        "programmers": "program",
        "programmed": "program",
        "designing": "design",
        "designer": "design",
        "designers": "design",
        "designed": "design",
        "architect": "architect",
        "architecture": "architect",
        "architectures": "architect",
        "specialist": "special",
        "specialized": "special",
        "specializing": "special",
        "scalable": "scale",
        "scalability": "scale",
        "scaling": "scale",
        "testing": "test",
        "tester": "test",
        "tested": "test",
        "integration": "integrat",
        "integrated": "integrat",
        "integrating": "integrat",
        "requirements": "require",
        "requirement": "require",
        "required": "require",
        "applications": "app",
        "application": "app",
        "apps": "app",
        "technologies": "technolog",
        "technology": "technolog",
        "technical": "technolog",
    }
    if w in irreg:
        return irreg[w]

    for suffix in ("ational", "tional", "ization", "isation", "fulness", "ousness", "iveness"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    for suffix in ("ement", "ment", "ness", "able", "ible", "ical", "ally", "ting", "ling", "ring", "ning", "zing"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    for suffix in ("ing", "ion", "ive", "ity", "ous", "ies", "ied", "ed", "es", "ly", "er", "al"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            base = w[:-len(suffix)]
            if len(base) > 2 and base.endswith(base[-1]) and base[-1] not in "lsz":
                base = base[:-1]
            return base
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def robust_tokenize(text: str) -> list:
    """
    Tokenize text into clean tokens preserving tech-specific terms:
    c++, c#, .net, node.js, rest apis, and single-letter languages like c, r.
    """
    if not text:
        return []
    s = text.lower()
    s = s.replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")
    tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", s)
    restored = []
    for t in tokens:
        if t == "cpp":
            restored.append("c++")
        elif t == "csharp":
            restored.append("c#")
        elif t == "dotnet":
            restored.append(".net")
        else:
            restored.append(t)
    return restored


def _smart_text_similarity(doc_text: str, query_text: str) -> float:
    """
    Compute multi-layered text similarity between document (resume) and query (job description).
    Uses stemmed term overlap, sublinear TF-IDF character & word N-grams, and length-adaptive scoring.
    """
    if not doc_text or not query_text:
        return 0.0

    t1 = doc_text.strip()
    t2 = query_text.strip()
    if not t1 or not t2:
        return 0.0

    doc_tokens = robust_tokenize(t1)
    query_tokens = robust_tokenize(t2)
    if not doc_tokens or not query_tokens:
        return 0.0

    # Meaningful query tokens (excluding pure stop words)
    meaningful_query = [t for t in query_tokens if t not in ALL_STOP_WORDS and len(t) > 1]
    if not meaningful_query:
        meaningful_query = [t for t in query_tokens if t not in STANDARD_STOP_WORDS and len(t) > 1]
    if not meaningful_query:
        meaningful_query = query_tokens

    # 1. Stemmed Token Coverage
    doc_stems = {simple_stem(t) for t in doc_tokens}
    query_stems = [simple_stem(t) for t in meaningful_query]
    matched_stems = [s for s in query_stems if s in doc_stems]
    stem_coverage = len(matched_stems) / len(query_stems) if query_stems else 0.0

    # 2. Character N-gram TF-IDF (robust to word variants and typos)
    char_cos = 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
        char_matrix = char_vec.fit_transform([t1, t2])
        char_cos = float(cosine_similarity(char_matrix[0:1], char_matrix[1:2])[0][0])
    except Exception:
        char_cos = 0.0

    # 3. Word N-gram TF-IDF
    word_cos = 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        word_vec = TfidfVectorizer(
            token_pattern=r"(?u)\S+",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        word_matrix = word_vec.fit_transform([" ".join(doc_tokens), " ".join(query_tokens)])
        word_cos = float(cosine_similarity(word_matrix[0:1], word_matrix[1:2])[0][0])
    except Exception:
        word_cos = 0.0

    # Scaling: In short query vs long document comparisons, cosine is naturally compressed
    scaled_char = min(1.0, (char_cos / 0.22) ** 0.8) if char_cos > 0 else 0.0
    scaled_word = min(1.0, (word_cos / 0.18) ** 0.8) if word_cos > 0 else 0.0

    # Composite similarity
    if stem_coverage > 0 or scaled_char > 0 or scaled_word > 0:
        composite = (
            0.45 * stem_coverage +
            0.30 * scaled_char +
            0.25 * scaled_word
        ) * 100.0
    else:
        composite = 0.0

    return round(float(np.clip(composite, 0.0, 100.0)), 1)


def compute_overall_similarity(
    resume_text: str,
    jd_text: str,
    resume_skills: list = None,
    jd_skills: list = None
) -> float:
    """
    Compute overall semantic similarity between resume and job description.
    Blends deep text similarity, character/word N-gram matching, and skill alignment.
    """
    if not resume_text or not jd_text:
        return 0.0

    text_score = 0.0
    model = _get_model()

    if model is not None:
        try:
            embeddings = model.encode([resume_text[:4000], jd_text[:4000]], show_progress_bar=False, convert_to_numpy=True)
            similarity = float(cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0])
            bert_scaled = max(0.0, min(100.0, ((similarity - 0.15) / 0.70) * 100))
            stat_score = _smart_text_similarity(resume_text[:5000], jd_text[:5000])
            text_score = 0.60 * bert_scaled + 0.40 * stat_score
        except Exception:
            text_score = _smart_text_similarity(resume_text[:5000], jd_text[:5000])
    else:
        text_score = _smart_text_similarity(resume_text[:5000], jd_text[:5000])

    # Blend with skill matching if available
    if resume_skills is not None and jd_skills is not None:
        skill_res = compute_skill_match(resume_skills, jd_skills, jd_text=jd_text)
        skill_pct = skill_res.get("match_percentage", 0.0)

        if len(jd_skills) > 0:
            combined = max(text_score, 0.50 * text_score + 0.50 * skill_pct)
        else:
            # If JD had no explicit skills extracted, text_score is primary
            combined = text_score
        return round(float(np.clip(combined, 0.0, 100.0)), 1)

    return round(float(np.clip(text_score, 0.0, 100.0)), 1)


def compute_section_scores(resume_sections: dict, jd_text: str) -> dict:
    """
    Compute relevance scores for each resume section against the JD.
    """
    scores = {}
    scoreable_sections = [
        "experience", "skills", "education", "projects",
        "summary", "certifications", "achievements",
    ]

    jd_truncated = jd_text[:3000]

    for section_name in scoreable_sections:
        if section_name in resume_sections:
            section_text = resume_sections[section_name][:2000]
            scores[section_name] = _smart_text_similarity(
                section_text, jd_truncated
            )

    return scores


def _skills_overlap(res_canon: str, jd_canon: str) -> bool:
    """
    Check if two canonical skill keys match directly, as substrings, or through common stems.
    """
    if res_canon == jd_canon:
        return True
    # Substring match for compound skills (e.g. 'rest' in 'rest apis', 'full stack' in 'full stack development')
    if (len(res_canon) >= 3 and len(jd_canon) >= 3) and (res_canon in jd_canon or jd_canon in res_canon):
        return True
    # Stemmed token overlap
    res_stems = {simple_stem(t) for t in res_canon.split() if t not in STANDARD_STOP_WORDS}
    jd_stems = {simple_stem(t) for t in jd_canon.split() if t not in STANDARD_STOP_WORDS}
    if res_stems and jd_stems and bool(res_stems & jd_stems):
        return True
    return False


def compute_skill_match(resume_skills: list, jd_skills: list, jd_text: str = "") -> dict:
    """
    Compute skill overlap between resume and job description.
    Uses canonical keys, alias resolution, and stem-based matching.
    """
    matched = []
    missing = []
    extra = []

    matched_jd_indices = set()
    matched_resume_indices = set()

    # Match each JD skill against resume skills
    for j_idx, jd_s in enumerate(jd_skills):
        jd_canon = jd_s.get("canonical", jd_s["skill"].lower())
        found_match = False

        for r_idx, res_s in enumerate(resume_skills):
            res_canon = res_s.get("canonical", res_s["skill"].lower())
            if _skills_overlap(res_canon, jd_canon):
                found_match = True
                matched_jd_indices.add(j_idx)
                matched_resume_indices.add(r_idx)
                matched.append(jd_s)
                break

        if not found_match:
            missing.append(jd_s)

    # Resume skills not matched to JD skills are bonus/extra
    for r_idx, res_s in enumerate(resume_skills):
        if r_idx not in matched_resume_indices:
            extra.append(res_s)

    # Sort
    missing.sort(key=lambda s: (0 if s.get("priority") == "required" else 1, s["skill"]))
    matched.sort(key=lambda s: s["skill"])
    extra.sort(key=lambda s: s["skill"])

    if jd_skills:
        match_percentage = round((len(matched) / len(jd_skills)) * 100, 1)
    else:
        # If JD had 0 detected skills, evaluate against JD raw text if available
        if jd_text and resume_skills:
            jd_lower = jd_text.lower()
            text_matched = [
                s for s in resume_skills
                if s.get("canonical", s["skill"].lower()) in jd_lower or simple_stem(s["skill"].lower()) in jd_lower
            ]
            if text_matched:
                match_percentage = round(min(100.0, (len(text_matched) / max(3, len(resume_skills))) * 100), 1)
                matched = text_matched
            else:
                match_percentage = 40.0  # Fair default when JD is too short/generic to specify skills
        else:
            match_percentage = 50.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_percentage": match_percentage,
        "total_jd_skills": len(jd_skills),
        "total_resume_skills": len(resume_skills),
    }
