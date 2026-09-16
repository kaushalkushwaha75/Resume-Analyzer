"""
Resume Analyzer — Flask Application
AI-powered resume evaluation against job descriptions.
"""

import os
import uuid
import traceback
from typing import cast

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
)

from werkzeug.utils import secure_filename

from config import Config

from analyzer.parser import parse_file

from analyzer.preprocessor import (
    clean_text,
    extract_sections,
)

from analyzer.skill_extractor import (
    extract_skills,
    extract_skills_from_jd,
    categorize_skills,
)

from analyzer.matcher import (
    compute_overall_similarity,
    compute_section_scores,
    compute_skill_match,
)

from analyzer.suggestion_engine import (
    generate_suggestions,
    compute_ats_score,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Load configuration
app.config.from_object(Config)

# Initialize configuration
Config.init_app(app)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    """
    Check whether uploaded file has an allowed extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in Config.ALLOWED_EXTENSIONS
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """
    Display the resume upload page.
    """

    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# RESUME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze uploaded resume against a job description.

    Analysis pipeline:

    1. Validate uploaded resume
    2. Validate job description
    3. Save uploaded file
    4. Parse PDF/DOCX
    5. Clean resume text
    6. Extract resume sections
    7. Extract resume skills
    8. Extract job-description skills
    9. Calculate semantic similarity
    10. Calculate section relevance
    11. Calculate skill matching
    12. Generate suggestions
    13. Calculate ATS score
    14. Calculate final composite score
    15. Display results
    """

    file_path = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GET UPLOADED FILE
    # ─────────────────────────────────────────────────────────────────────────

    file = request.files.get("resume")

    # Get job description
    jd_text = request.form.get(
        "job_description",
        ""
    ).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VALIDATE FILE
    # ─────────────────────────────────────────────────────────────────────────

    if file is None:
        flash(
            "No resume file uploaded.",
            "error"
        )
        return redirect(url_for("index"))

    # Check filename
    if not file.filename:
        flash(
            "No file selected.",
            "error"
        )
        return redirect(url_for("index"))

    # Tell Pylance that filename is definitely a string
    original_filename = cast(
        str,
        file.filename
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. VALIDATE FILE TYPE
    # ─────────────────────────────────────────────────────────────────────────

    if not allowed_file(original_filename):
        flash(
            "Invalid file type. Please upload a PDF or DOCX file.",
            "error"
        )
        return redirect(url_for("index"))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. VALIDATE JOB DESCRIPTION
    # ─────────────────────────────────────────────────────────────────────────

    if not jd_text:
        flash(
            "Please paste a job description.",
            "error"
        )
        return redirect(url_for("index"))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CREATE SAFE FILE NAME
    # ─────────────────────────────────────────────────────────────────────────

    filename = secure_filename(
        original_filename
    )

    # Generate unique filename
    unique_name = (
        f"{uuid.uuid4().hex[:8]}_{filename}"
    )

    # Get upload folder from configuration
    upload_folder = app.config["UPLOAD_FOLDER"]

    # Create upload folder if it doesn't exist
    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    # Full file path
    file_path = os.path.join(
        upload_folder,
        unique_name
    )

    try:

        # ═══════════════════════════════════════════════════════════════════════
        # SAVE FILE
        # ═══════════════════════════════════════════════════════════════════════

        file.save(file_path)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1 — PARSE DOCUMENT
        # ═══════════════════════════════════════════════════════════════════════

        raw_text = parse_file(
            file_path
        )

        # Check extracted text
        if not raw_text or not raw_text.strip():

            flash(
                "Could not extract text from the uploaded resume.",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2 — CLEAN TEXT
        # ═══════════════════════════════════════════════════════════════════════

        cleaned_text = clean_text(
            raw_text
        )

        # Check minimum resume length
        if len(cleaned_text) < Config.MIN_RESUME_LENGTH:

            flash(
                "The uploaded file contains too little text. "
                "Please upload a valid resume.",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3 — EXTRACT RESUME SECTIONS
        # ═══════════════════════════════════════════════════════════════════════

        sections = extract_sections(
            cleaned_text
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4 — EXTRACT SKILLS
        # ═══════════════════════════════════════════════════════════════════════

        # Skills found in resume
        resume_skills = extract_skills(
            cleaned_text
        )

        cleaned_jd = clean_text(jd_text)

        # Skills found in job description
        jd_skills = extract_skills_from_jd(
            cleaned_jd
        )

        # Categorize resume skills
        resume_skills_categorized = categorize_skills(
            resume_skills
        )

        # Categorize JD skills
        jd_skills_categorized = categorize_skills(
            jd_skills
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5 — SEMANTIC SIMILARITY
        # ═══════════════════════════════════════════════════════════════════════

        overall_score = compute_overall_similarity(
            cleaned_text,
            cleaned_jd,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6 — SECTION RELEVANCE
        # ═══════════════════════════════════════════════════════════════════════

        section_scores = compute_section_scores(
            sections,
            cleaned_jd
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 7 — SKILL MATCHING
        # ═══════════════════════════════════════════════════════════════════════

        skill_match = compute_skill_match(
            resume_skills,
            jd_skills,
            jd_text=cleaned_jd
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 8 — PREPARE ANALYSIS DATA
        # ═══════════════════════════════════════════════════════════════════════

        analysis_result = {

            "overall_score": overall_score,

            "skill_match": skill_match,

            "section_scores": section_scores,

            "sections": sections,

            "resume_text": cleaned_text,

            "jd_text": jd_text,
        }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 9 — GENERATE SUGGESTIONS
        # ═══════════════════════════════════════════════════════════════════════

        suggestions = generate_suggestions(
            analysis_result
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 10 — ATS SCORE
        # ═══════════════════════════════════════════════════════════════════════

        ats_result = compute_ats_score(
            cleaned_text,
            sections
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 11 — FINAL COMPOSITE SCORE
        # ═══════════════════════════════════════════════════════════════════════
        #
        # Semantic Similarity = 40%
        # Skill Match         = 35%
        # ATS Score            = 25%
        #
        # Total                = 100%
        # ═══════════════════════════════════════════════════════════════════════

        composite_score = round(
            (
                overall_score * 0.40
                +
                skill_match["match_percentage"] * 0.35
                +
                ats_result["percentage"] * 0.25
            ),
            1
        )

        # Keep score between 0 and 100
        composite_score = max(
            0.0,
            min(
                100.0,
                composite_score
            )
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 12 — PREPARE RESULTS
        # ═══════════════════════════════════════════════════════════════════════

        results = {

            # Overall score
            "composite_score": composite_score,

            # Semantic similarity
            "overall_similarity": overall_score,

            # Skill analysis
            "skill_match": skill_match,

            # Resume skills
            "resume_skills": resume_skills,

            # JD skills
            "jd_skills": jd_skills,

            # Categorized resume skills
            "resume_skills_categorized": (
                resume_skills_categorized
            ),

            # Categorized JD skills
            "jd_skills_categorized": (
                jd_skills_categorized
            ),

            # Section scores
            "section_scores": section_scores,

            # Sections detected
            "sections_found": [
                key
                for key in sections.keys()
                if key != "full_text"
            ],

            # Suggestions
            "suggestions": suggestions,

            # ATS result
            "ats": ats_result,

            # Word count
            "resume_word_count": len(
                cleaned_text.split()
            ),

            # Original filename
            "filename": filename,
        }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 13 — SHOW RESULTS
        # ═══════════════════════════════════════════════════════════════════════

        return render_template(
            "results.html",
            results=results
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ERROR HANDLING
    # ═══════════════════════════════════════════════════════════════════════════

    except ValueError as e:

        flash(
            str(e),
            "error"
        )

        return redirect(
            url_for("index")
        )

    except Exception as e:

        # Print complete error in terminal
        traceback.print_exc()

        flash(
            f"An error occurred during analysis: {str(e)}",
            "error"
        )

        return redirect(
            url_for("index")
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # DELETE UPLOADED FILE
    # ═══════════════════════════════════════════════════════════════════════════

    finally:

        if file_path and os.path.exists(file_path):

            try:

                os.remove(file_path)

            except OSError:

                pass


# ═══════════════════════════════════════════════════════════════════════════════
# JSON API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    JSON API endpoint for resume analysis.

    Accepts:
        resume
        job_description
    """

    file_path = None

    # ─────────────────────────────────────────────────────────────────────────
    # GET INPUT
    # ─────────────────────────────────────────────────────────────────────────

    file = request.files.get("resume")

    jd_text = request.form.get(
        "job_description",
        ""
    ).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATE FILE
    # ─────────────────────────────────────────────────────────────────────────

    if file is None:

        return jsonify(
            {
                "error": "No resume file provided"
            }
        ), 400

    if not file.filename:

        return jsonify(
            {
                "error": "No resume file selected"
            }
        ), 400

    # Convert filename to a definite string
    original_filename = cast(
        str,
        file.filename
    )

    # Validate extension
    if not allowed_file(original_filename):

        return jsonify(
            {
                "error": (
                    "Invalid file. "
                    "Upload PDF or DOCX."
                )
            }
        ), 400

    # Validate JD
    if not jd_text:

        return jsonify(
            {
                "error": "Job description is required."
            }
        ), 400

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE FILE
    # ─────────────────────────────────────────────────────────────────────────

    filename = secure_filename(
        original_filename
    )

    unique_name = (
        f"{uuid.uuid4().hex[:8]}_{filename}"
    )

    upload_folder = app.config["UPLOAD_FOLDER"]

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        upload_folder,
        unique_name
    )

    try:

        file.save(file_path)

        # ═══════════════════════════════════════════════════════════════════════
        # PARSE
        # ═══════════════════════════════════════════════════════════════════════

        raw_text = parse_file(
            file_path
        )

        if not raw_text or not raw_text.strip():

            return jsonify(
                {
                    "error": (
                        "Could not extract text "
                        "from resume."
                    )
                }
            ), 400

        # ═══════════════════════════════════════════════════════════════════════
        # CLEAN TEXT
        # ═══════════════════════════════════════════════════════════════════════

        cleaned_text = clean_text(
            raw_text
        )

        if len(cleaned_text) < Config.MIN_RESUME_LENGTH:

            return jsonify(
                {
                    "error": (
                        "Resume contains too little text."
                    )
                }
            ), 400

        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACT SECTIONS
        # ═══════════════════════════════════════════════════════════════════════

        sections = extract_sections(
            cleaned_text
        )

        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACT SKILLS
        # ═══════════════════════════════════════════════════════════════════════

        resume_skills = extract_skills(
            cleaned_text
        )

        cleaned_jd = clean_text(jd_text)

        jd_skills = extract_skills_from_jd(
            cleaned_jd
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SEMANTIC MATCH
        # ═══════════════════════════════════════════════════════════════════════

        overall_score = compute_overall_similarity(
            cleaned_text,
            cleaned_jd,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION MATCH
        # ═══════════════════════════════════════════════════════════════════════

        section_scores = compute_section_scores(
            sections,
            cleaned_jd
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SKILL MATCH
        # ═══════════════════════════════════════════════════════════════════════

        skill_match = compute_skill_match(
            resume_skills,
            jd_skills
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SUGGESTIONS
        # ═══════════════════════════════════════════════════════════════════════

        analysis_result = {

            "overall_score": overall_score,

            "skill_match": skill_match,

            "section_scores": section_scores,

            "sections": sections,

            "resume_text": cleaned_text,

            "jd_text": jd_text,
        }

        suggestions = generate_suggestions(
            analysis_result
        )

        # ═══════════════════════════════════════════════════════════════════════
        # ATS SCORE
        # ═══════════════════════════════════════════════════════════════════════

        ats_result = compute_ats_score(
            cleaned_text,
            sections
        )

        # ═══════════════════════════════════════════════════════════════════════
        # COMPOSITE SCORE
        # ═══════════════════════════════════════════════════════════════════════

        composite_score = round(
            (
                overall_score * 0.40
                +
                skill_match["match_percentage"] * 0.35
                +
                ats_result["percentage"] * 0.25
            ),
            1
        )

        composite_score = max(
            0.0,
            min(
                100.0,
                composite_score
            )
        )

        # ═══════════════════════════════════════════════════════════════════════
        # JSON RESPONSE
        # ═══════════════════════════════════════════════════════════════════════

        return jsonify(
            {

                "composite_score": (
                    composite_score
                ),

                "overall_similarity": (
                    overall_score
                ),

                "skill_match": (
                    skill_match
                ),

                "section_scores": (
                    section_scores
                ),

                "ats": (
                    ats_result
                ),

                "suggestions": (
                    suggestions
                ),

                "resume_skills": (
                    resume_skills
                ),

                "jd_skills": (
                    jd_skills
                ),

                "resume_word_count": len(
                    cleaned_text.split()
                ),
            }
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # API ERROR HANDLING
    # ═══════════════════════════════════════════════════════════════════════════

    except ValueError as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 400

    except Exception as e:

        traceback.print_exc()

        return jsonify(
            {
                "error": (
                    f"Analysis failed: {str(e)}"
                )
            }
        ), 500

    # ═══════════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════════

    finally:

        if file_path and os.path.exists(file_path):

            try:

                os.remove(file_path)

            except OSError:

                pass


# ═══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print()
    print("[*] Resume Analyzer starting...")
    print("    Visit: http://127.0.0.1:5000/")
    print()

    app.run(
        debug=Config.DEBUG,
        host="127.0.0.1",
        port=5000,
    )
    