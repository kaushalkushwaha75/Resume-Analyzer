"""
Text preprocessor for resumes and job descriptions.

Handles cleaning, normalization, section extraction,
tokenization, and basic entity extraction.

This version does NOT require spaCy.
"""

import re


# Common resume section headers
SECTION_PATTERNS = {
    "contact": r"(?i)^(?:contact\s*(?:info(?:rmation)?)?|personal\s*(?:info(?:rmation)?|details)?)",
    "summary": r"(?i)^(?:summary|objective|profile|about\s*me|professional\s*summary|career\s*(?:summary|objective))",
    "experience": r"(?i)^(?:(?:work|professional|employment)\s*(?:experience|history)|experience|work\s*history)",
    "education": r"(?i)^(?:education(?:al)?\s*(?:background|qualifications?)?|academic\s*(?:background|qualifications?)?|qualifications?)",
    "skills": r"(?i)^(?:(?:technical\s*|core\s*|key\s*)?skills|competenc(?:ies|e)|technologies|tech\s*stack|areas?\s*of\s*expertise)",
    "projects": r"(?i)^(?:projects?|(?:personal|academic|key)\s*projects?|portfolio)",
    "certifications": r"(?i)^(?:certifications?|licenses?\s*(?:&|and)\s*certifications?|professional\s*certifications?)",
    "achievements": r"(?i)^(?:achievements?|accomplishments?|awards?\s*(?:&|and)\s*(?:achievements?|honors?)|honors?)",
    "publications": r"(?i)^(?:publications?|research|papers?)",
    "languages": r"(?i)^(?:languages?|language\s*proficiency)",
    "interests": r"(?i)^(?:interests?|hobbies?|extracurricular)",
    "references": r"(?i)^(?:references?|referees?)",
}


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from a document.
    """

    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize bullet characters
    text = re.sub(
        r"[•●■◦▪▸►‣⁃–—]",
        "- ",
        text
    )

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Collapse spaces/tabs but preserve new lines
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_sections(text: str) -> dict:
    """
    Split resume text into logical sections based on common headers.
    """

    sections = {"full_text": text}

    if not text:
        return sections

    lines = text.split("\n")

    current_section = "header"
    current_content = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            current_content.append("")
            continue

        matched_section = None

        # Check whether current line is a section heading
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, stripped):
                matched_section = section_name
                break

        if matched_section:

            # Save previous section
            if current_content:
                content = "\n".join(current_content).strip()

                if content:
                    sections[current_section] = content

            current_section = matched_section
            current_content = []

        else:
            current_content.append(line)

    # Save final section
    if current_content:
        content = "\n".join(current_content).strip()

        if content:
            sections[current_section] = content

    return sections


def tokenize(text: str) -> list:
    """
    Basic tokenizer without spaCy.

    Converts text to lowercase and extracts words.
    Common stopwords are removed.
    """

    if not text:
        return []

    # Convert to lowercase
    text = text.lower()

    # Extract alphabetic words
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    # Basic English stopwords
    stopwords = {
        "the", "and", "for", "with", "this", "that",
        "from", "are", "was", "were", "have", "has",
        "had", "will", "would", "can", "could", "should",
        "you", "your", "our", "their", "they", "them",
        "his", "her", "its", "into", "about", "over",
        "under", "after", "before", "between", "through",
        "during", "using", "used", "use", "also", "very",
        "more", "most", "some", "such", "than", "then",
        "who", "what", "when", "where", "which", "while",
        "how", "why", "not", "but", "all", "any", "both",
        "each", "other", "only", "own", "same", "too",
        "our", "out", "off", "been", "being", "doing",
        "does", "did", "get", "got", "make", "made"
    }

    tokens = [
        word
        for word in words
        if word not in stopwords
    ]

    return tokens


def extract_entities(text: str) -> dict:
    """
    Basic entity extraction without spaCy.

    Detects common resume entities such as:
    - Email
    - Phone
    - URLs
    """

    if not text:
        return {}

    entities = {}

    # Email addresses
    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text
    )

    if emails:
        entities["EMAIL"] = list(dict.fromkeys(emails))

    # Phone numbers
    phones = re.findall(
        r"(?:\+91[-\s]?)?[6-9]\d{9}\b",
        text
    )

    if phones:
        entities["PHONE"] = list(dict.fromkeys(phones))

    # URLs
    urls = re.findall(
        r"https?://[^\s]+|www\.[^\s]+",
        text
    )

    if urls:
        entities["URL"] = list(dict.fromkeys(urls))

    return entities