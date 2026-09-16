import os

class Config:
    """Application configuration."""

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "resume-analyzer-dev-key-change-in-prod")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # File uploads
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB max file size
    ALLOWED_EXTENSIONS = {"pdf", "docx"}

    # NLP Models
    SPACY_MODEL = "en_core_web_sm"
    BERT_MODEL = "all-MiniLM-L6-v2"

    # Analysis
    MIN_RESUME_LENGTH = 50  # Minimum characters for a valid resume
    MAX_SUGGESTIONS = 15  # Max improvement suggestions to show

    @staticmethod
    def init_app(app):
        """Initialize application with config."""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
