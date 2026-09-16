"""
Skill extraction engine.
Maintains a curated skills database and extracts skills from text
using pattern matching, canonical aliases, and dynamic keyword extraction.
"""

import re
from typing import Optional, Dict, List, Set


# ─── Curated Skills Database ────────────────────────────────────────────────────
# Organized by category with extensive synonyms, aliases, and role titles.

SKILLS_DATABASE = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c", "c++", "cpp", "c#", "csharp",
        "ruby", "go", "golang", "rust", "swift", "kotlin", "scala", "php",
        "r", "matlab", "perl", "lua", "dart", "elixir", "haskell",
        "objective-c", "assembly", "fortran", "cobol", "visual basic", "vb.net",
        "shell", "bash", "powershell", "sql", "nosql", "plsql", "t-sql",
    ],
    "Computer Science & Core": [
        "data structures", "algorithms", "dsa", "data structures and algorithms",
        "oop", "oops", "object oriented programming", "object-oriented programming",
        "operating systems", "os", "system design", "distributed systems",
        "computer networks", "networking", "dbms", "database management",
        "database management systems", "multithreading", "concurrency",
        "debugging", "problem solving", "clean code", "design patterns",
        "computer science", "software engineering", "software development",
    ],
    "Roles & Methodologies": [
        "software engineer", "software developer", "full stack developer",
        "full stack development", "full stack", "frontend developer",
        "frontend development", "frontend", "backend developer",
        "backend development", "backend", "web developer", "web development",
        "mobile developer", "app development", "systems engineer", "programmer",
        "coding", "agile", "scrum", "kanban", "sprint planning", "sdlc",
        "test driven development", "tdd", "devops",
    ],
    "Web Frameworks & Stacks": [
        "react", "reactjs", "react.js", "angular", "angularjs", "vue",
        "vuejs", "vue.js", "next.js", "nextjs", "nuxt", "nuxtjs",
        "svelte", "django", "flask", "fastapi", "express", "expressjs",
        "spring", "spring boot", "rails", "ruby on rails", "laravel",
        "asp.net", ".net", "dotnet", "gatsby", "remix", "ember",
        "mern", "mern stack", "mean", "mean stack", "lamp stack", "node", "nodejs", "node.js",
    ],
    "Mobile Development": [
        "react native", "flutter", "swift", "swiftui", "kotlin",
        "android", "ios", "xamarin", "ionic", "cordova",
    ],
    "Databases & Storage": [
        "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite",
        "oracle", "sql server", "dynamodb", "cassandra", "couchdb",
        "neo4j", "elasticsearch", "mariadb", "firebase", "supabase",
        "cockroachdb", "influxdb", "relational database", "vector database",
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
        "ci/cd", "ci cd", "github actions", "gitlab ci", "circleci", "travis ci",
        "heroku", "vercel", "netlify", "cloudflare", "nginx", "apache",
        "linux", "ubuntu", "centos", "helm", "prometheus", "grafana",
        "datadog", "new relic", "vagrant", "serverless",
    ],
    "Data Science & AI": [
        "machine learning", "deep learning", "artificial intelligence", "ai",
        "neural networks", "natural language processing", "nlp",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "plotly", "jupyter", "hugging face", "transformers", "bert",
        "gpt", "llm", "llms", "large language models", "generative ai", "genai",
        "rag", "retrieval augmented generation", "speech-to-text", "stt",
        "opencv", "spacy", "nltk", "xgboost", "lightgbm", "random forest",
        "data mining", "feature engineering", "model deployment",
        "mlops", "mlflow", "kubeflow", "airflow", "data pipeline",
    ],
    "Data Engineering & Analytics": [
        "etl", "data warehouse", "data lake", "spark", "apache spark",
        "hadoop", "hive", "kafka", "apache kafka", "flink", "beam",
        "bigquery", "redshift", "snowflake", "databricks", "dbt",
        "tableau", "power bi", "looker", "metabase", "superset",
        "excel", "google sheets", "data visualization", "data analysis",
        "business intelligence", "statistics", "a/b testing",
    ],
    "Frontend & UI/UX": [
        "html", "html5", "css", "css3", "sass", "scss", "less", "tailwind",
        "tailwindcss", "bootstrap", "material ui", "mui", "chakra ui",
        "ant design", "figma", "sketch", "adobe xd", "photoshop", "illustrator",
        "responsive design", "responsive web design", "ui/ux", "user experience",
        "user interface", "accessibility", "wcag", "seo", "webpack", "vite", "babel",
        "storybook", "jest", "cypress", "playwright", "selenium",
    ],
    "Version Control & Tools": [
        "git", "github", "gitlab", "bitbucket", "svn", "jira", "confluence",
        "trello", "asana", "slack", "notion", "vs code", "visual studio code",
    ],
    "Cybersecurity": [
        "cybersecurity", "penetration testing", "ethical hacking",
        "owasp", "encryption", "ssl", "tls", "oauth", "jwt", "sso",
        "identity management", "iam", "siem", "firewall", "vpn",
        "incident response", "vulnerability assessment", "soc",
    ],
    "APIs & Microservices": [
        "rest", "rest api", "rest apis", "restful", "restful api", "restful apis",
        "graphql", "grpc", "websocket", "soap", "api design", "microservices",
        "event-driven", "message queue", "rabbitmq", "celery", "swagger", "openapi", "postman",
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "time management", "project management",
        "presentation", "mentoring", "collaboration", "adaptability",
        "creativity", "analytical thinking", "decision making",
        "conflict resolution", "negotiation", "public speaking",
        "interpersonal skills", "organizational skills", "multitasking",
        "attention to detail", "self-motivated", "team player",
        "cross-functional", "stakeholder management",
    ],
}

# Canonical normalization mapping: maps any variation/synonym to (Canonical Display Name, Canonical Key)
CANONICAL_MAP: Dict[str, tuple] = {
    # CS & Core
    "dsa": ("Data Structures & Algorithms", "data structures & algorithms"),
    "data structures": ("Data Structures & Algorithms", "data structures & algorithms"),
    "algorithms": ("Data Structures & Algorithms", "data structures & algorithms"),
    "data structures and algorithms": ("Data Structures & Algorithms", "data structures & algorithms"),
    "oop": ("OOP (Object-Oriented Programming)", "oop"),
    "oops": ("OOP (Object-Oriented Programming)", "oop"),
    "object oriented programming": ("OOP (Object-Oriented Programming)", "oop"),
    "object-oriented programming": ("OOP (Object-Oriented Programming)", "oop"),
    "operating systems": ("Operating Systems", "operating systems"),
    "os": ("Operating Systems", "operating systems"),
    "dbms": ("DBMS (Database Management)", "dbms"),
    "database management": ("DBMS (Database Management)", "dbms"),
    "database management systems": ("DBMS (Database Management)", "dbms"),
    "debugging": ("Debugging", "debugging"),
    "problem solving": ("Problem Solving", "problem solving"),
    "clean code": ("Clean Code", "clean code"),
    "system design": ("System Design", "system design"),
    "computer science": ("Computer Science", "computer science"),
    "software engineering": ("Software Engineering", "software engineering"),
    "software development": ("Software Engineering", "software engineering"),
    "software engineer": ("Software Engineering", "software engineering"),
    "software developer": ("Software Engineering", "software engineering"),
    "full stack": ("Full Stack Development", "full stack"),
    "full stack developer": ("Full Stack Development", "full stack"),
    "full stack development": ("Full Stack Development", "full stack"),
    "frontend": ("Frontend Development", "frontend"),
    "frontend developer": ("Frontend Development", "frontend"),
    "frontend development": ("Frontend Development", "frontend"),
    "backend": ("Backend Development", "backend"),
    "backend developer": ("Backend Development", "backend"),
    "backend development": ("Backend Development", "backend"),
    "web developer": ("Web Development", "web development"),
    "web development": ("Web Development", "web development"),
    "developer": ("Software Development", "developer"),
    "engineer": ("Software Engineering", "engineer"),
    "programmer": ("Software Development", "developer"),
    "coding": ("Programming", "programming"),
    # Web & Stacks
    "html": ("HTML5", "html"),
    "html5": ("HTML5", "html"),
    "css": ("CSS3", "css"),
    "css3": ("CSS3", "css"),
    "react": ("React", "react"),
    "reactjs": ("React", "react"),
    "react.js": ("React", "react"),
    "angular": ("Angular", "angular"),
    "angularjs": ("Angular", "angular"),
    "vue": ("Vue", "vue"),
    "vuejs": ("Vue", "vue"),
    "vue.js": ("Vue", "vue"),
    "next.js": ("Next.js", "next.js"),
    "nextjs": ("Next.js", "next.js"),
    "node": ("Node.js", "node.js"),
    "nodejs": ("Node.js", "node.js"),
    "node.js": ("Node.js", "node.js"),
    "express": ("Express", "express"),
    "expressjs": ("Express", "express"),
    "mern": ("MERN Stack", "mern stack"),
    "mern stack": ("MERN Stack", "mern stack"),
    "mean": ("MEAN Stack", "mean stack"),
    "mean stack": ("MEAN Stack", "mean stack"),
    "fastapi": ("FastAPI", "fastapi"),
    "spring boot": ("Spring Boot", "spring boot"),
    "spring": ("Spring", "spring"),
    "django": ("Django", "django"),
    "flask": ("Flask", "flask"),
    # APIs
    "rest": ("REST APIs", "rest apis"),
    "rest api": ("REST APIs", "rest apis"),
    "rest apis": ("REST APIs", "rest apis"),
    "restful": ("REST APIs", "rest apis"),
    "restful api": ("REST APIs", "rest apis"),
    "restful apis": ("REST APIs", "rest apis"),
    # AI & ML
    "ai": ("Artificial Intelligence", "ai"),
    "artificial intelligence": ("Artificial Intelligence", "ai"),
    "machine learning": ("Machine Learning", "machine learning"),
    "deep learning": ("Deep Learning", "deep learning"),
    "nlp": ("NLP", "nlp"),
    "natural language processing": ("NLP", "nlp"),
    "speech-to-text": ("Speech-to-Text", "speech-to-text"),
    "stt": ("Speech-to-Text", "speech-to-text"),
    "rag": ("RAG (Retrieval-Augmented Generation)", "rag"),
    "retrieval augmented generation": ("RAG (Retrieval-Augmented Generation)", "rag"),
    "llm": ("LLMs", "llm"),
    "llms": ("LLMs", "llm"),
    "large language models": ("LLMs", "llm"),
    "generative ai": ("Generative AI", "generative ai"),
    "genai": ("Generative AI", "generative ai"),
    # Databases
    "sql": ("SQL", "sql"),
    "mysql": ("MySQL", "mysql"),
    "postgresql": ("PostgreSQL", "postgresql"),
    "postgres": ("PostgreSQL", "postgresql"),
    "mongodb": ("MongoDB", "mongodb"),
    # Languages
    "python": ("Python", "python"),
    "javascript": ("JavaScript", "javascript"),
    "typescript": ("TypeScript", "typescript"),
    "java": ("Java", "java"),
    "c++": ("C++", "c++"),
    "cpp": ("C++", "c++"),
    "c#": ("C#", "c#"),
    "csharp": ("C#", "c#"),
    "c": ("C", "c"),
    # Tools
    "git": ("Git", "git"),
    "github": ("GitHub", "github"),
    "vs code": ("VS Code", "vs code"),
    "visual studio code": ("VS Code", "vs code"),
    "responsive design": ("Responsive Web Design", "responsive web design"),
    "responsive web design": ("Responsive Web Design", "responsive web design"),
}

# Build a flat lookup: skill_phrase (lowercase) -> category
_SKILL_LOOKUP: Dict[str, str] = {}
for category, skills in SKILLS_DATABASE.items():
    for skill in skills:
        _SKILL_LOOKUP[skill.lower()] = category

# Sort skills by length descending so longer multi-word phrases match before sub-parts
_SORTED_SKILLS = sorted(_SKILL_LOOKUP.keys(), key=len, reverse=True)


def get_canonical(skill_name: str) -> tuple:
    """
    Return (display_name, canonical_key) for a skill name.
    """
    clean = skill_name.strip().lower()
    if clean in CANONICAL_MAP:
        return CANONICAL_MAP[clean]
    # Default display formatting
    if len(clean) <= 3 and not clean.endswith("."):
        display = clean.upper()
    else:
        display = clean.title()
    return display, clean


def extract_skills(text: str) -> List[Dict]:
    """
    Extract skills from text using pattern matching against the skills database.
    Normalizes duplicates to their canonical forms.

    Returns:
        List of dicts: [
            {
                "skill": str (Display Name),
                "canonical": str (Comparison Key),
                "category": str,
                "confidence": float,
                "count": int
            }
        ]
    """
    if not text:
        return []

    text_lower = text.lower()
    found_canonicals = set()
    found_skills = []

    for skill in _SORTED_SKILLS:
        display_name, canonical_key = get_canonical(skill)
        if canonical_key in found_canonicals:
            continue

        # Build word-boundary regex for the skill
        escaped = re.escape(skill)
        # Handle special tech punctuation like c++, c#, .net
        if skill in ("c++", "c#", "c", "r", "go"):
            pattern = r"(?<![a-zA-Z0-9_])" + escaped + r"(?![a-zA-Z0-9_+#])"
        elif skill.startswith("."):
            pattern = r"(?<![a-zA-Z0-9_])" + escaped + r"(?![a-zA-Z0-9_])"
        else:
            pattern = r"(?<![a-zA-Z0-9_])" + escaped + r"(?![a-zA-Z0-9_])"

        matches = re.findall(pattern, text_lower)
        if matches:
            count = len(matches)
            confidence = min(1.0, 0.65 + (count - 1) * 0.15)
            category = _SKILL_LOOKUP.get(skill, "Technical Skills")

            found_skills.append({
                "skill": display_name,
                "canonical": canonical_key,
                "category": category,
                "confidence": round(confidence, 2),
                "count": count,
            })
            found_canonicals.add(canonical_key)

    # Sort by confidence descending, then alphabetically
    found_skills.sort(key=lambda s: (-s["confidence"], s["skill"]))
    return found_skills


def extract_dynamic_jd_keywords(text: str) -> List[Dict]:
    """
    Dynamic keyword and requirement extractor for job descriptions.
    Finds technical terms, roles, and requirement concepts even if they are
    not present in the static skills database.
    """
    if not text:
        return []

    text_clean = text.strip()
    extracted = []
    seen = set()

    # 1. Role / Title patterns: e.g. "Software Developer", "Full Stack Engineer", "Developer"
    role_pattern = r"\b(?:([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+)?(developer|engineer|specialist|architect|programmer|lead|intern)\b"
    stopwords_prefix = {"the", "a", "an", "our", "for", "as", "and", "or", "looking", "seeking", "hire", "hiring", "need", "needed", "join", "team", "with", "to", "in"}
    for match in re.finditer(role_pattern, text_clean, re.IGNORECASE):
        prefix = (match.group(1) or "").strip().lower()
        role_noun = match.group(2).strip().lower()
        clean_prefixes = [p for p in prefix.split() if p not in stopwords_prefix]
        if clean_prefixes:
            role = " ".join(clean_prefixes) + " " + role_noun
        else:
            role = role_noun
        disp, canon = get_canonical(role)
        if canon not in seen and len(canon) >= 3:
            seen.add(canon)
            extracted.append({
                "skill": disp,
                "canonical": canon,
                "category": "Role & Responsibilities",
                "confidence": 0.85,
                "count": 1,
                "priority": "required"
            })

    # 2. Acronyms & Tech abbreviations (2-5 uppercase letters)
    tech_acronyms = set(re.findall(r"\b[A-Z]{2,5}\b", text_clean))
    ignore_acronyms = {"THE", "AND", "FOR", "WITH", "NOT", "ALL", "YOU", "ARE", "HAS", "HAD", "CAN", "MAY"}
    for ac in tech_acronyms:
        if ac not in ignore_acronyms:
            ac_lower = ac.lower()
            if ac_lower not in seen:
                disp, canon = get_canonical(ac)
                seen.add(canon)
                extracted.append({
                    "skill": disp,
                    "canonical": canon,
                    "category": "Technical Skills",
                    "confidence": 0.80,
                    "count": 1,
                    "priority": "required"
                })

    # 3. Required / Proficiency phrases: e.g. "Proficiency in X", "Experience in Y"
    req_phrases = re.findall(
        r"(?:proficiency in|experience (?:in|with)|knowledge of|skilled in|understanding of)\s+([A-Za-z0-9+#\.\s]{2,30}?)(?:[,;\.\n]|and\b)",
        text_clean,
        re.IGNORECASE
    )
    for phrase in req_phrases:
        cleaned_phrase = phrase.strip()
        phrase_lower = cleaned_phrase.lower()
        if len(cleaned_phrase) >= 2 and phrase_lower not in seen and len(cleaned_phrase.split()) <= 4:
            disp, canon = get_canonical(cleaned_phrase)
            seen.add(canon)
            extracted.append({
                "skill": disp,
                "canonical": canon,
                "category": "Core Requirements",
                "confidence": 0.80,
                "count": 1,
                "priority": "required"
            })

    return extracted


def extract_skills_from_jd(text: str) -> List[Dict]:
    """
    Extract required and preferred skills from a job description.
    Uses curated database extraction, and if few/no skills are detected,
    supplements with dynamic requirement extraction so JD skills are never empty.
    """
    if not text:
        return []

    # 1. Base curated extraction
    skills = extract_skills(text)
    seen_canonicals = {s.get("canonical", s["skill"].lower()) for s in skills}

    # 2. If fewer than 3 skills found, augment with dynamic extraction
    if len(skills) < 3:
        dynamic_skills = extract_dynamic_jd_keywords(text)
        for ds in dynamic_skills:
            if ds["canonical"] not in seen_canonicals:
                skills.append(ds)
                seen_canonicals.add(ds["canonical"])

    # Determine priority based on surrounding context
    text_lower = text.lower()
    required_patterns = [
        r"(?:required|must\s+have|essential|mandatory|minimum|need)",
        r"(?:requirements?|qualifications?)\s*:",
    ]
    preferred_patterns = [
        r"(?:preferred|nice\s+to\s+have|bonus|plus|desired|optional|advantageous)",
    ]

    for skill_info in skills:
        skill_clean = skill_info.get("canonical", skill_info["skill"].lower())

        is_required = False
        is_preferred = False

        for pattern in required_patterns:
            search_pattern = pattern + r".{0,200}" + re.escape(skill_clean[:15])
            if re.search(search_pattern, text_lower, re.DOTALL):
                is_required = True
                break

        if not is_required:
            for pattern in preferred_patterns:
                search_pattern = pattern + r".{0,200}" + re.escape(skill_clean[:15])
                if re.search(search_pattern, text_lower, re.DOTALL):
                    is_preferred = True
                    break

        skill_info["priority"] = (
            "required" if is_required
            else "preferred" if is_preferred
            else "required"
        )

    return skills


def categorize_skills(skills: list) -> dict:
    """
    Group a list of extracted skills by their category.
    """
    categorized = {}
    for skill_info in skills:
        cat = skill_info.get("category", "Other Skills")
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append(skill_info)

    return categorized


def get_all_categories() -> list:
    """Return all skill category names."""
    return list(SKILLS_DATABASE.keys())
