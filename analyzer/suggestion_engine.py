"""
Suggestion engine for resume improvement.
Generates categorized, actionable suggestions based on analysis results.
Also includes ATS compatibility scoring.
"""

import re


# ─── Power Verbs ────────────────────────────────────────────────────────────────
POWER_VERBS = {
    "leadership": ["spearheaded", "directed", "orchestrated", "championed", "pioneered"],
    "achievement": ["achieved", "exceeded", "surpassed", "delivered", "accomplished"],
    "technical": ["engineered", "architected", "developed", "implemented", "optimized"],
    "collaboration": ["collaborated", "partnered", "coordinated", "facilitated", "contributed"],
    "analysis": ["analyzed", "evaluated", "assessed", "investigated", "researched"],
    "improvement": ["improved", "enhanced", "streamlined", "revamped", "transformed"],
    "creation": ["created", "designed", "built", "established", "launched"],
    "management": ["managed", "oversaw", "supervised", "administered", "maintained"],
}

# Common weak verbs to flag
WEAK_VERBS = [
    "helped", "assisted", "worked on", "was responsible for",
    "participated in", "involved in", "handled", "did", "made",
]

# Common section headers an ATS expects
ATS_EXPECTED_SECTIONS = [
    "experience", "education", "skills", "summary",
]


def generate_suggestions(analysis_result: dict) -> list:
    """
    Generate categorized improvement suggestions based on analysis.

    Args:
        analysis_result: Dict containing all analysis data:
            - overall_score: float (0-100)
            - skill_match: dict from matcher.compute_skill_match()
            - section_scores: dict from matcher.compute_section_scores()
            - sections: dict of detected resume sections
            - resume_text: full resume text
            - jd_text: full job description text

    Returns:
        List of suggestion dicts:
        [{"severity": str, "category": str, "title": str, "detail": str}]
        severity: "critical" | "important" | "nice-to-have"
    """
    suggestions = []
    skill_match = analysis_result.get("skill_match", {})
    section_scores = analysis_result.get("section_scores", {})
    sections = analysis_result.get("sections", {})
    resume_text = analysis_result.get("resume_text", "")
    overall_score = analysis_result.get("overall_score", 0)

    # ── Critical Suggestions ──────────────────────────────────────────────────

    # 1. Missing required skills
    missing = skill_match.get("missing", [])
    required_missing = [s for s in missing if s.get("priority") == "required"]
    if required_missing:
        skill_names = ", ".join(s["skill"] for s in required_missing[:8])
        suggestions.append({
            "severity": "critical",
            "category": "Skills Gap",
            "title": f"Missing {len(required_missing)} required skill(s) from the job description",
            "detail": f"The job description requires: **{skill_names}**. "
                      f"Add these skills to your resume if you have experience with them, "
                      f"or consider learning them. Place them in a dedicated 'Skills' section.",
        })

    # 2. Very low overall match
    if overall_score < 30:
        suggestions.append({
            "severity": "critical",
            "category": "Overall Fit",
            "title": "Very low match with the job description",
            "detail": "Your resume content has minimal overlap with this job description. "
                      "Consider tailoring your resume specifically for this role by using "
                      "similar keywords, highlighting relevant projects, and rewriting your "
                      "summary to align with the job requirements.",
        })

    # 3. Missing critical sections
    for section in ["experience", "education", "skills"]:
        if section not in sections:
            suggestions.append({
                "severity": "critical",
                "category": "Missing Section",
                "title": f"Missing '{section.title()}' section",
                "detail": f"Your resume does not appear to have a clearly labeled "
                          f"'{section.title()}' section. Most ATS systems and recruiters "
                          f"expect this section. Add a clear section header.",
            })

    # ── Important Suggestions ─────────────────────────────────────────────────

    # 4. Missing preferred skills
    preferred_missing = [s for s in missing if s.get("priority") == "preferred"]
    if preferred_missing:
        skill_names = ", ".join(s["skill"] for s in preferred_missing[:6])
        suggestions.append({
            "severity": "important",
            "category": "Skills Enhancement",
            "title": f"{len(preferred_missing)} preferred skill(s) not found",
            "detail": f"These skills are preferred by the employer: **{skill_names}**. "
                      f"Including them could strengthen your application.",
        })

    # 5. Low section scores
    for section_name, score in section_scores.items():
        if score < 40 and section_name in sections:
            suggestions.append({
                "severity": "important",
                "category": "Section Relevance",
                "title": f"'{section_name.title()}' section has low relevance ({score:.0f}%)",
                "detail": f"Your {section_name} section doesn't strongly align with the "
                          f"job description. Consider adding more relevant keywords and "
                          f"experiences that match the role requirements.",
            })

    # 6. Weak verbs detection
    text_lower = resume_text.lower()
    found_weak_verbs = [v for v in WEAK_VERBS if v in text_lower]
    if found_weak_verbs:
        weak_list = ", ".join(f'"{v}"' for v in found_weak_verbs[:5])
        suggestions.append({
            "severity": "important",
            "category": "Language Strength",
            "title": "Weak action verbs detected",
            "detail": f"Your resume uses weak verbs like {weak_list}. "
                      f"Replace them with stronger power verbs such as "
                      f"'spearheaded', 'engineered', 'achieved', 'optimized', "
                      f"or 'delivered' to make a stronger impression.",
        })

    # 7. Missing summary/objective
    if "summary" not in sections:
        suggestions.append({
            "severity": "important",
            "category": "Missing Section",
            "title": "No professional summary found",
            "detail": "Adding a 2-3 sentence professional summary at the top of your "
                      "resume helps recruiters quickly understand your background and "
                      "career goals. Tailor it to match the job description.",
        })

    # 8. Resume length check
    word_count = len(resume_text.split())
    if word_count < 150:
        suggestions.append({
            "severity": "important",
            "category": "Content Depth",
            "title": "Resume appears too short",
            "detail": f"Your resume contains approximately {word_count} words. "
                      f"Most effective resumes have 300-700 words. Consider adding "
                      f"more detail about your experiences and achievements.",
        })
    elif word_count > 1200:
        suggestions.append({
            "severity": "important",
            "category": "Content Length",
            "title": "Resume may be too long",
            "detail": f"Your resume contains approximately {word_count} words. "
                      f"For most positions (especially entry-level), a one-page resume "
                      f"(300-700 words) is preferred. Consider condensing less relevant sections.",
        })

    # ── Nice-to-Have Suggestions ──────────────────────────────────────────────

    # 9. Quantifiable achievements
    has_numbers = bool(re.search(r'\d+[%$]|\$\d+|\d+\s*(?:users?|clients?|projects?|team)', text_lower))
    if not has_numbers:
        suggestions.append({
            "severity": "nice-to-have",
            "category": "Impact Metrics",
            "title": "Add quantifiable achievements",
            "detail": "Recruiters love numbers. Instead of 'Improved website performance', "
                      "try 'Improved website load time by 40%, serving 10K+ daily users'. "
                      "Add metrics like percentages, dollar amounts, team sizes, or user counts.",
        })

    # 10. Missing certifications section
    if "certifications" not in sections:
        suggestions.append({
            "severity": "nice-to-have",
            "category": "Credentials",
            "title": "Consider adding a Certifications section",
            "detail": "If you hold any relevant certifications (AWS, Google Cloud, "
                      "Scrum Master, etc.), adding a Certifications section can "
                      "strengthen your resume significantly.",
        })

    # 11. Missing projects section (especially for students)
    if "projects" not in sections:
        suggestions.append({
            "severity": "nice-to-have",
            "category": "Portfolio",
            "title": "Consider adding a Projects section",
            "detail": "For students and early-career professionals, a Projects section "
                      "is an excellent way to demonstrate practical skills. Include "
                      "personal projects, hackathon entries, or academic projects with "
                      "links to GitHub repositories.",
        })

    # 12. Power verb suggestions
    suggestions.append({
        "severity": "nice-to-have",
        "category": "Writing Tips",
        "title": "Use strong action verbs",
        "detail": "Start each bullet point with a powerful action verb. Examples: "
                  "'Engineered a real-time data pipeline...', "
                  "'Spearheaded the migration to microservices...', "
                  "'Achieved 99.9% uptime through proactive monitoring...'",
    })

    return suggestions


def compute_ats_score(resume_text: str, sections: dict) -> dict:
    """
    Compute ATS (Applicant Tracking System) compatibility score.

    Args:
        resume_text: Full resume text.
        sections: Detected sections dict.

    Returns:
        Dict with overall ATS score and individual checks.
    """
    checks = []
    total_score = 0
    max_score = 0

    # 1. Section headers present (25 points)
    max_score += 25
    found_sections = [s for s in ATS_EXPECTED_SECTIONS if s in sections]
    section_ratio = len(found_sections) / len(ATS_EXPECTED_SECTIONS)
    section_score = round(25 * section_ratio)
    total_score += section_score
    checks.append({
        "name": "Section Headers",
        "score": section_score,
        "max": 25,
        "passed": section_ratio >= 0.75,
        "detail": f"Found {len(found_sections)}/{len(ATS_EXPECTED_SECTIONS)} expected sections: "
                  f"{', '.join(s.title() for s in found_sections) if found_sections else 'None'}",
    })

    # 2. Contact information (20 points)
    max_score += 20
    text_lower = resume_text.lower()
    has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', resume_text))
    has_phone = bool(re.search(r'[\+]?[\d\s\-\(\)]{7,15}', resume_text))
    has_linkedin = "linkedin" in text_lower
    contact_items = sum([has_email, has_phone, has_linkedin])
    contact_score = min(20, contact_items * 7)
    total_score += contact_score
    checks.append({
        "name": "Contact Information",
        "score": contact_score,
        "max": 20,
        "passed": contact_items >= 2,
        "detail": f"{'✅' if has_email else '❌'} Email | "
                  f"{'✅' if has_phone else '❌'} Phone | "
                  f"{'✅' if has_linkedin else '❌'} LinkedIn",
    })

    # 3. Keyword density / sufficient content (20 points)
    max_score += 20
    word_count = len(resume_text.split())
    if word_count >= 300:
        density_score = 20
    elif word_count >= 150:
        density_score = 12
    else:
        density_score = 5
    total_score += density_score
    checks.append({
        "name": "Content Depth",
        "score": density_score,
        "max": 20,
        "passed": word_count >= 250,
        "detail": f"Resume contains ~{word_count} words. "
                  f"{'Good length.' if word_count >= 300 else 'Consider adding more detail.'}",
    })

    # 4. Clean formatting (15 points)
    max_score += 15
    has_special_chars = bool(re.search(r'[★☆✦✧◆◇▲△●○■□]', resume_text))
    has_excessive_caps = len(re.findall(r'\b[A-Z]{5,}\b', resume_text)) > 5
    has_urls = bool(re.search(r'https?://\S+', resume_text))
    format_issues = sum([has_special_chars, has_excessive_caps])
    format_score = max(0, 15 - format_issues * 5)
    total_score += format_score
    checks.append({
        "name": "Formatting",
        "score": format_score,
        "max": 15,
        "passed": format_issues == 0,
        "detail": f"{'Clean formatting ✅' if format_issues == 0 else 'Some formatting issues detected — avoid special symbols and excessive caps.'}",
    })

    # 5. Consistency & professionalism (20 points)
    max_score += 20
    has_dates = bool(re.search(r'(?:19|20)\d{2}', resume_text))
    has_bullets = bool(re.search(r'(?:^|\n)\s*[-•]', resume_text))
    has_action_verbs = any(
        verb in text_lower
        for verbs in POWER_VERBS.values()
        for verb in verbs
    )
    prof_items = sum([has_dates, has_bullets, has_action_verbs])
    prof_score = min(20, prof_items * 7)
    total_score += prof_score
    checks.append({
        "name": "Professionalism",
        "score": prof_score,
        "max": 20,
        "passed": prof_items >= 2,
        "detail": f"{'✅' if has_dates else '❌'} Date formatting | "
                  f"{'✅' if has_bullets else '❌'} Bullet points | "
                  f"{'✅' if has_action_verbs else '❌'} Action verbs",
    })

    return {
        "score": total_score,
        "max_score": max_score,
        "percentage": round((total_score / max_score) * 100, 1) if max_score > 0 else 0,
        "checks": checks,
    }
