import base64
import os
import re
import requests
import streamlit as st
from openai import OpenAI

# ==========================================
# Configuration & Constants
# ==========================================
GH = "https://api.github.com"
DEFAULT_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL = "alibaba-qwen3-32b"

SUPPORTED_MODELS = [
    "alibaba-qwen3-32b",
    "meta-llama/llama-3.1-70b-instruct",
    "deepseek-v3.2",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Custom..."
]

SAMPLE_REPOS = [
    ("Nexora-AI", "https://github.com/coder-Yash886/Nexora-AI"),
    ("FastAPI", "https://github.com/fastapi/fastapi"),
    ("Flask", "https://github.com/pallets/flask"),
    ("Requests", "https://github.com/psf/requests")
]

# ==========================================
# Helper Functions & .env Loader
# ==========================================
def load_dotenv_file(filepath=".env"):
    """Load key-value pairs from .env into os.environ."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k:
                            os.environ[k] = v
        except Exception:
            pass

load_dotenv_file()

def secret(name, default=""):
    """Fetch secret from session state, .streamlit/secrets, or .env."""
    if f"ui_{name}" in st.session_state and st.session_state[f"ui_{name}"]:
        val = str(st.session_state[f"ui_{name}"]).strip()
        if val and val != "your_key_here":
            return val
    try:
        if name in st.secrets:
            val = str(st.secrets[name]).strip()
            if val and val != "your_key_here":
                return val
    except Exception:
        pass
    val = os.getenv(name, default)
    if isinstance(val, str):
        val = val.strip()
        if val and val != "your_key_here":
            return val
    return default

def gh_get(path, params=None):
    """Make authenticated or unauthenticated GET request to GitHub API."""
    headers = {"Accept": "application/vnd.github+json"}
    token = secret("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        r = requests.get(GH + path, headers=headers, params=params, timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error connecting to GitHub: {e}")

    if r.status_code == 404:
        raise RuntimeError("Repository not found, or it is private. Please verify the URL.")
    if r.status_code == 403:
        raise RuntimeError("GitHub rate limit reached (60/hr). Add a GITHUB_TOKEN in settings to lift limit to 5000/hr.")
    r.raise_for_status()
    return r.json()

def parse_repo(url):
    """Extract owner and repository name from GitHub URL."""
    url = url.strip()
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        raise ValueError("Please enter a valid GitHub repository URL (e.g., https://github.com/owner/repo).")
    return m.group(1), m.group(2).removesuffix(".git")

def fetch_repo(owner, repo):
    """Fetch repo metadata, README, recursive tree, and beginner issues."""
    info = gh_get(f"/repos/{owner}/{repo}")
    
    readme = ""
    try:
        data = gh_get(f"/repos/{owner}/{repo}/readme")
        readme = base64.b64decode(data["content"]).decode("utf-8", "ignore")
    except Exception:
        pass

    branch = info.get("default_branch", "main")
    paths = []
    try:
        tree = gh_get(f"/repos/{owner}/{repo}/git/trees/{branch}", {"recursive": "1"})
        skip = (".git", "node_modules", "vendor", "dist", "build", ".github", "__pycache__")
        paths = [
            t["path"] for t in tree.get("tree", [])
            if t["type"] == "blob" and not t["path"].startswith(skip)
        ]
    except Exception:
        pass

    issues = []
    try:
        issues_raw = gh_get(
            f"/repos/{owner}/{repo}/issues",
            {"labels": "good first issue", "state": "open", "per_page": 6},
        )
        issues = [i for i in issues_raw if "pull_request" not in i]
    except Exception:
        pass

    if not issues:
        try:
            issues_raw = gh_get(
                f"/repos/{owner}/{repo}/issues",
                {"state": "open", "per_page": 6},
            )
            issues = [i for i in issues_raw if "pull_request" not in i]
        except Exception:
            pass

    return info, readme, paths, issues

def fetch_issue_body(owner, repo, issue_number):
    """Fetch complete issue details including description and labels."""
    try:
        return gh_get(f"/repos/{owner}/{repo}/issues/{issue_number}")
    except Exception:
        return {}

def classify_file_tree(paths):
    """Categorize repository file paths for intuitive architecture browsing."""
    categories = {
        "Source & Logic": [],
        "Configuration & Build": [],
        "Tests & Quality": [],
        "Documentation & Community": [],
        "Other Assets": []
    }
    
    code_exts = ('.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.c', '.cpp', '.rb', '.php', '.cs', '.html', '.css')
    config_names = ('docker', 'compose', 'package.json', 'requirements', 'pyproject', 'cargo', 'gemfile', 'makefile', 'setup.py', '.yaml', '.yml', '.toml', '.env')
    
    for p in paths:
        low = p.lower()
        if 'test' in low or 'spec' in low:
            categories["Tests & Quality"].append(p)
        elif 'doc' in low or 'readme' in low or 'contributing' in low or 'license' in low:
            categories["Documentation & Community"].append(p)
        elif any(cfg in low for cfg in config_names):
            categories["Configuration & Build"].append(p)
        elif low.endswith(code_exts):
            categories["Source & Logic"].append(p)
        else:
            categories["Other Assets"].append(p)
            
    return categories

def ask_llm(prompt, model=None, base_url=None, api_key=None):
    """Query open-weight LLM on DigitalOcean Gradient Serverless Inference."""
    key = api_key or secret("DO_MODEL_ACCESS_KEY")
    url = base_url or secret("DO_BASE_URL", DEFAULT_BASE_URL)
    mod = model or secret("DO_MODEL_ID", DEFAULT_MODEL)

    if not key:
        return None

    try:
        client = OpenAI(
            api_key=key,
            base_url=url,
        )
        resp = client.chat.completions.create(
            model=mod,
            messages=[
                {"role": "system", "content": "You are a friendly, encouraging open-source mentor for beginners."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return cleaned
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "403" in err_msg:
            raise RuntimeError("DigitalOcean Authentication failed. Please verify your DO_MODEL_ACCESS_KEY.")
        elif "404" in err_msg:
            raise RuntimeError(f"Model ID '{mod}' not found on DigitalOcean. Check the Model Catalog.")
        else:
            raise RuntimeError(f"DigitalOcean Inference Error: {err_msg}")

def generate_smart_guide(info, readme, paths, issues, lang):
    """Generate an intelligent beginner-friendly onboarding guide grounded in actual repo data."""
    repo_name = info.get('name', 'Project')
    full_name = info.get('full_name', repo_name)
    desc = info.get('description') or f"Open-source software project built with {info.get('language', 'modern technologies')}."
    lang_name = info.get('language') or 'General'
    
    if lang == "Hinglish":
        intro = f"**{repo_name}** ek powerful open-source project hai: {desc}.\nYeh project primary taur par **{lang_name}** mein build kiya gaya hai aur beginners ke liye explore aur contribute karne ke liye structured hai."
    elif lang == "Hindi":
        intro = f"**{repo_name}** एक बेहतरीन ओपन-सोर्स प्रोजेक्ट है। {desc}\nयह मुख्य रूप से **{lang_name}** भाषा में बनाया गया है और डेवलपर्स के लिए बहुत उपयोगी है।"
    else:
        intro = f"**{repo_name}** is a modern open-source project: {desc}.\nIt is primarily engineered in **{lang_name}**, built to be modular, efficient, and developer-friendly."

    tree_categories = classify_file_tree(paths)
    key_files = []
    if tree_categories["Source & Logic"]:
        key_files.append(f"- **`{tree_categories['Source & Logic'][0]}`**: Core codebase logic and implementation")
        if len(tree_categories["Source & Logic"]) > 1:
            key_files.append(f"- **`{tree_categories['Source & Logic'][1]}`**: Module handlers & utility functions")
    if tree_categories["Configuration & Build"]:
        key_files.append(f"- **`{tree_categories['Configuration & Build'][0]}`**: Project dependencies and build setup")
    if tree_categories["Tests & Quality"]:
        key_files.append(f"- **`{tree_categories['Tests & Quality'][0]}`**: Test suites to verify codebase integrity")
    if tree_categories["Documentation & Community"]:
        key_files.append(f"- **`{tree_categories['Documentation & Community'][0]}`**: Guidelines, documentation, and specs")
    
    if not key_files:
        key_files = [f"- `{p}`: Repository root file" for p in paths[:5]]
    
    map_text = "\n".join(key_files)

    run_steps = []
    run_steps.append(f"1. **Clone the repository:**\n   ```bash\n   git clone https://github.com/{full_name}.git\n   cd {repo_name}\n   ```")
    
    if "python" in lang_name.lower() or any("requirements" in p.lower() or "pyproject" in p.lower() for p in paths):
        run_steps.append("2. **Set up virtual environment & install dependencies:**\n   ```bash\n   python -m venv venv\n   source venv/bin/activate  # Windows: venv\\Scripts\\activate\n   pip install -r requirements.txt\n   ```")
        run_steps.append("3. **Run the application / tests:**\n   ```bash\n   python app.py  # or pytest\n   ```")
    elif "javascript" in lang_name.lower() or "typescript" in lang_name.lower() or any("package.json" in p.lower() for p in paths):
        run_steps.append("2. **Install Node packages:**\n   ```bash\n   npm install\n   ```")
        run_steps.append("3. **Start local dev server:**\n   ```bash\n   npm run dev\n   ```")
    else:
        run_steps.append("2. **Follow setup instructions:**\n   Refer to `README.md` for build and launch instructions.")
    
    local_text = "\n".join(run_steps)

    contrib_points = []
    if issues:
        first_issue = issues[0]
        contrib_points.append(f"1. **Start with open Issue #{first_issue['number']}**: [{first_issue['title']}]({first_issue['html_url']})")
    else:
        contrib_points.append("1. **Documentation fix**: Check README or contributing docs for typos or missing instructions.")
    contrib_points.append("2. **Add unit test cases**: Check the test directory and write a test for uncovered edge cases.")
    contrib_points.append("3. **Create your feature branch & submit PR**: `git checkout -b fix/issue-name` and create a PR!")
    
    contrib_text = "\n".join(contrib_points)

    return f"""## What this project does
{intro}

- **Repository**: `{full_name}`
- **Primary Language**: {lang_name}
- **Community**: ⭐ {info.get('stargazers_count', 0):,} stars | 🍴 {info.get('forks_count', 0):,} forks

## Where things live
A bird's-eye architectural map of where components live:
{map_text}

## How to run it locally
{local_text}

## Where to start contributing
{contrib_text}"""

# ==========================================
# Streamlit UI Configuration
# ==========================================
st.set_page_config(
    page_title="RepoGuide — Open-Source AI Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Premium Design System (Plus Jakarta Sans + Emerald Theme)
# ==========================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Smooth Scroll Behavior & Anchor Offsets */
    html {
        scroll-behavior: smooth !important;
    }
    [id] {
        scroll-margin-top: 100px !important;
    }

    /* Remove default link underline and colors */
    a, a:visited, a:active {
        text-decoration: none !important;
    }

    /* Remove default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Layout Centering & Responsive Canvas */
    .block-container {
        max-width: 1140px !important;
        padding-top: 85px !important;
        padding-bottom: 4rem !important;
        margin: 0 auto !important;
    }
    
    /* Background with soft subtle emerald mesh glow */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FFFFFF !important;
        background-image: radial-gradient(circle at 10% 8%, rgba(16, 185, 129, 0.08) 0%, transparent 40%), radial-gradient(circle at 90% 75%, rgba(16, 185, 129, 0.04) 0%, transparent 35%) !important;
        background-attachment: fixed !important;
        color: #0F172A !important;
    }

    /* Fixed Floating Glassmorphic Top Navbar */
    .top-sticky-nav {
        position: fixed !important;
        top: 12px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(1140px, calc(100% - 32px)) !important;
        z-index: 999999 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 12px 24px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.08) !important;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        text-decoration: none !important;
    }
    .brand-logo-icon {
        width: 36px;
        height: 36px;
        background: #10B981;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .nav-links {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .top-sticky-nav a.nav-item {
        color: #475569 !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }
    .top-sticky-nav a.nav-item:hover {
        color: #10B981 !important;
        background-color: #ECFDF5 !important;
    }
    .btn-nav-primary {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        padding: 9px 20px !important;
        border-radius: 9999px !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .btn-nav-primary:hover {
        background-color: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    }

    /* Section Spacing & Anchors */
    .section-wrapper {
        padding: 55px 0 22px 0;
        border-top: 1px solid #F1F5F9;
    }
    .section-tag {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .section-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin-bottom: 10px;
    }
    .section-desc {
        color: #64748B;
        font-size: 1.05rem;
        max-width: 640px;
        margin-bottom: 30px;
        line-height: 1.6;
    }

    /* Hero Headline & Elements */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        border-radius: 9999px;
        padding: 6px 16px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 18px;
    }
    .hero-heading {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.035em;
        color: #0F172A;
        margin-bottom: 18px;
    }
    .hero-green-text {
        color: #10B981;
    }
    .hero-desc {
        color: #475569;
        font-size: 1.1rem;
        line-height: 1.65;
        margin-bottom: 24px;
        max-width: 560px;
    }

    /* Hero CTA Button Group */
    .hero-btn-group {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 22px;
        flex-wrap: wrap;
    }
    .btn-hero-primary {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        padding: 13px 26px !important;
        border-radius: 9999px !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .btn-hero-primary:hover {
        background-color: #059669 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(16, 185, 129, 0.45) !important;
    }
    .btn-hero-secondary {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #E2E8F0 !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 9999px !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }
    .btn-hero-secondary:hover {
        border-color: #10B981 !important;
        color: #059669 !important;
        background-color: #F8FAFC !important;
        transform: translateY(-1px) !important;
    }
    .hero-subpoints {
        display: flex;
        align-items: center;
        gap: 18px;
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 600;
        flex-wrap: wrap;
    }
    .hero-subpoint-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .hero-check-icon {
        color: #10B981;
        font-weight: 800;
    }

    /* Modern Mockup Frame */
    .mockup-container {
        position: relative;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.02);
    }
    .mockup-inner {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
    }
    .mockup-floating-card {
        background: #0F172A;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        margin-top: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }

    /* Feature Cards */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    .feature-card:hover {
        border-color: #10B981;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.12);
        transform: translateY(-2px);
    }
    .feature-icon-box {
        width: 44px;
        height: 44px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #059669;
        font-size: 22px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .feature-text {
        font-size: 0.92rem;
        color: #64748B;
        line-height: 1.6;
    }

    /* Step Timeline Cards */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .step-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #10B981;
        opacity: 0.9;
        margin-bottom: 8px;
    }

    /* Inputs & Buttons Styling */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1.5px solid #CBD5E1 !important;
        padding: 11px 16px !important;
        font-size: 0.95rem !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2) !important;
    }
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1.5px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
    }

    .stButton button[kind="primary"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 11px 24px !important;
        font-size: 1rem !important;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
    }

    .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #334155 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="secondary"]:hover {
        border-color: #10B981 !important;
        color: #059669 !important;
        background-color: #ECFDF5 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #F1F5F9 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #059669 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }

    /* Result Cards */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
    }
    .issue-card-clean {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .issue-card-clean:hover {
        border-color: #10B981;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.1);
        transform: translateY(-1px);
    }

    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #ECFDF5 !important;
        color: #059669 !important;
        padding: 2px 6px !important;
        border-radius: 5px !important;
        font-size: 0.88em !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STICKY TOP NAVBAR (Smooth Anchor Navigation)
# ==========================================
st.markdown("""
<div class="top-sticky-nav">
    <a href="#hero" target="_self" class="nav-brand">
        <div class="brand-logo-icon">✨</div>
        <span style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">RepoGuide</span>
    </a>
    <div class="nav-links">
        <a href="#hero" target="_self" class="nav-item">Home</a>
        <a href="#demo" target="_self" class="nav-item">Demo & Analyzer</a>
        <a href="#features" target="_self" class="nav-item">Features</a>
        <a href="#how-it-works" target="_self" class="nav-item">How it works</a>
        <a href="#architecture" target="_self" class="nav-item">Architecture</a>
        <a href="https://github.com/coder-Yash886/OpenPath" target="_blank" class="nav-item">GitHub</a>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <span style="font-size: 0.88rem; color: #64748B; font-weight: 600;">DigitalOcean AI</span>
        <a href="#demo" target="_self" class="btn-nav-primary">
            Try Demo &rarr;
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. HERO SECTION (#hero)
# ==========================================
st.markdown('<div id="hero"></div>', unsafe_allow_html=True)

col_hero_left, col_hero_right = st.columns([1.15, 0.85], gap="large")

with col_hero_left:
    st.markdown("""
    <div class="hero-badge">
        <span>⚡</span>
        <span>AI-powered open-source onboarding</span>
    </div>
    <div class="hero-heading">
        Contribute smarter with<br>
        <span class="hero-green-text">custom AI agents</span>
    </div>
    <p class="hero-desc">
        RepoGuide combines DigitalOcean Gradient AI, recursive code mapping, and live issue triage so your team and beginners understand any codebase in under a minute—not hours.
    </p>
    <div class="hero-btn-group">
        <a href="#demo" target="_self" class="btn-hero-primary">Try Demo &rarr;</a>
        <a href="#features" target="_self" class="btn-hero-secondary">Explore Features</a>
    </div>
    <div class="hero-subpoints">
        <span class="hero-subpoint-item"><span class="hero-check-icon">✓</span> 100% Open-Source AI</span>
        <span class="hero-subpoint-item"><span class="hero-check-icon">✓</span> Hinglish, Hindi, English</span>
        <span class="hero-subpoint-item"><span class="hero-check-icon">✓</span> DigitalOcean Gradient AI</span>
    </div>
    """, unsafe_allow_html=True)

with col_hero_right:
    st.markdown("""
    <div class="mockup-container">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #EF4444; display: inline-block;"></span>
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #F59E0B; display: inline-block;"></span>
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
            </div>
            <span style="font-size: 0.78rem; font-weight: 700; color: #10B981; background: #ECFDF5; padding: 3px 12px; border-radius: 9999px;">● DigitalOcean Inference Active</span>
        </div>
        <div class="mockup-inner">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">🧭 RepoGuide Dashboard</span>
                <span style="font-size: 0.75rem; color: #64748B;">v2.0 • Gradient AI</span>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; margin-bottom: 8px;">
                <div style="color: #059669; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span>✓</span> <span>Repository Map Generated</span>
                </div>
                <div style="color: #64748B; margin-top: 2px;">250 files parsed • 4 core modules mapped</div>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; font-size: 0.82rem;">
                <div style="color: #0284C7; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span>🎯</span> <span>Live Beginner Issues</span>
                </div>
                <div style="color: #64748B; margin-top: 2px;">Open 'good first issue' tickets detected</div>
            </div>
        </div>
        <div class="mockup-floating-card">
            <span>✨</span>
            <span>AI Mentor: Ready to guide your first Pull Request</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 4 Value Metric Highlights
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
hl_1, hl_2, hl_3, hl_4 = st.columns(4)
with hl_1:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; text-align: center;">
        <div style="font-size: 1.8rem; font-weight: 800; color: #10B981;">100%</div>
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">Open-Source AI</div>
    </div>
    """, unsafe_allow_html=True)
with hl_2:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; text-align: center;">
        <div style="font-size: 1.8rem; font-weight: 800; color: #0284C7;">3 Languages</div>
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">Hinglish, Hindi, English</div>
    </div>
    """, unsafe_allow_html=True)
with hl_3:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; text-align: center;">
        <div style="font-size: 1.8rem; font-weight: 800; color: #8B5CF6;">&lt; 60 Sec</div>
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">First PR Onboarding</div>
    </div>
    """, unsafe_allow_html=True)
with hl_4:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; text-align: center;">
        <div style="font-size: 1.8rem; font-weight: 800; color: #F59E0B;">DigitalOcean</div>
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">Serverless Inference</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 2. INTERACTIVE DEMO & LIVE ANALYZER (#demo)
# ==========================================
st.markdown("""
<div class="section-wrapper" id="demo">
    <div class="section-tag">Interactive Workspace</div>
    <div class="section-title">🚀 Live Repository Analyzer</div>
    <div class="section-desc">Paste any public GitHub repository URL below. Get an instant, structured onboarding guide, recursive file tree, live beginner issues, and ready-to-copy Git commands!</div>
</div>
""", unsafe_allow_html=True)

# One-Click Sample Repos
st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #475569; margin-bottom: 8px;'>⚡ One-Click Test Repositories:</p>", unsafe_allow_html=True)
c_cols = st.columns(len(SAMPLE_REPOS))
for idx, (label, s_url) in enumerate(SAMPLE_REPOS):
    if c_cols[idx].button(f"📦 {label}", key=f"chip_live_{idx}", use_container_width=True):
        st.session_state["target_repo_url"] = s_url
        st.session_state["auto_run_demo"] = True
        st.rerun()

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Main Search Input
cur_url = st.session_state.get("target_repo_url", "")
col_d_url, col_d_lang = st.columns([3, 1])
with col_d_url:
    demo_url = st.text_input(
        "GitHub repository URL",
        value=cur_url,
        placeholder="https://github.com/owner/repository",
        label_visibility="collapsed",
        key="main_demo_url_input"
    )
with col_d_lang:
    demo_lang = st.selectbox(
        "Language",
        ["Hinglish", "Hindi", "English"],
        index=0,
        label_visibility="collapsed",
        key="main_demo_lang_select"
    )

col_d_btn, col_d_set = st.columns([1.2, 2.8])
with col_d_btn:
    run_demo_button = st.button("🚀 Explain this repo", type="primary", use_container_width=True, key="btn_main_run")

with col_d_set:
    with st.expander("⚙️ DigitalOcean AI & Cloud Credentials Settings", expanded=False):
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            ui_do_key = st.text_input(
                "DigitalOcean Model Access Key",
                value=secret("DO_MODEL_ACCESS_KEY"),
                type="password",
                help="Your Model Access Key from DigitalOcean Inference Hub / Gradient AI.",
                key="ui_DO_MODEL_ACCESS_KEY"
            )
            model_choice = st.selectbox(
                "Open-Source Model",
                options=SUPPORTED_MODELS,
                index=0,
                key="model_select"
            )
            if model_choice == "Custom...":
                active_model = st.text_input("Custom Model ID", value="alibaba-qwen3-32b", key="ui_DO_MODEL_ID")
            else:
                active_model = model_choice
                st.session_state["ui_DO_MODEL_ID"] = active_model

        with c_k2:
            ui_base_url = st.text_input(
                "Inference Base URL",
                value=secret("DO_BASE_URL", DEFAULT_BASE_URL),
                key="ui_DO_BASE_URL"
            )
            ui_gh_token = st.text_input(
                "GitHub Token (Optional)",
                value=secret("GITHUB_TOKEN"),
                type="password",
                key="ui_GITHUB_TOKEN"
            )
        st.caption("✨ *Runs automatically in Smart Local Analyzer mode if no key is entered.*")

# Trigger Run Logic
trigger_run = run_demo_button or st.session_state.get("auto_run_demo", False)
if st.session_state.get("auto_run_demo", False):
    st.session_state["auto_run_demo"] = False

if trigger_run and demo_url:
    try:
        owner, repo = parse_repo(demo_url)
        with st.spinner(f"🔍 Reading repository '{owner}/{repo}' via GitHub API..."):
            info, readme, paths, issues = fetch_repo(owner, repo)

        st.session_state["current_info"] = info
        st.session_state["current_readme"] = readme
        st.session_state["current_paths"] = paths
        st.session_state["current_issues"] = issues
        st.session_state["current_owner"] = owner
        st.session_state["current_repo"] = repo
        st.session_state["current_lang"] = demo_lang
        st.session_state["repo_chat_history"] = []

        active_key = secret("DO_MODEL_ACCESS_KEY")
        base_url_to_use = secret("DO_BASE_URL", DEFAULT_BASE_URL)
        model_to_use = secret("DO_MODEL_ID", DEFAULT_MODEL)

        if active_key:
            with st.spinner(f"🧠 Calling DigitalOcean Gradient AI ({model_to_use})..."):
                issue_text = "\n".join(
                    f"#{i['number']} {i['title']} ({i['html_url']})" for i in issues
                ) or "No open 'good first issue' tickets found."

                prompt = f"""You are a friendly open-source mentor for beginners.
Explain this GitHub repository in simple {demo_lang}.

Repo: {info['full_name']}
Description: {info.get('description')}
Main language: {info.get('language')}

README (truncated):
{readme[:6000]}

File list (truncated):
{chr(10).join(paths[:250])}

Open good-first-issues:
{issue_text}

Reply in Markdown with exactly these sections:
## What this project does
(3-5 simple lines)

## Where things live
(a short map: important folders/files and what each one does)

## How to run it locally
(only steps supported by the README; say if unclear)

## Where to start contributing
(pick up to 3 beginner tasks, using the issues above if any)

Do not invent files that are not in the file list."""

                try:
                    answer = ask_llm(
                        prompt=prompt,
                        model=model_to_use,
                        base_url=base_url_to_use,
                        api_key=active_key
                    )
                    st.session_state["is_cloud_ai"] = True
                except Exception as llm_err:
                    st.warning(f"⚠️ DigitalOcean Cloud Inference notice: {llm_err}. Using Smart Analyzer fallback.")
                    answer = generate_smart_guide(info, readme, paths, issues, demo_lang)
                    st.session_state["is_cloud_ai"] = False
        else:
            with st.spinner("⚡ Analyzing codebase via RepoGuide Smart Engine..."):
                answer = generate_smart_guide(info, readme, paths, issues, demo_lang)
                st.session_state["is_cloud_ai"] = False

        st.session_state["current_guide"] = answer

    except ValueError as ve:
        st.warning(f"⚠️ {ve}")
    except RuntimeError as re_err:
        st.error(f"❌ {re_err}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

# Display Tabs and Result if Analyzed
if "current_guide" in st.session_state and st.session_state["current_guide"]:
    guide_content = st.session_state["current_guide"]
    issues = st.session_state.get("current_issues", [])
    owner = st.session_state.get("current_owner", "")
    repo = st.session_state.get("current_repo", "")
    info = st.session_state.get("current_info", {})
    paths = st.session_state.get("current_paths", [])
    readme = st.session_state.get("current_readme", "")
    lang = st.session_state.get("current_lang", "English")
    is_cloud_ai = st.session_state.get("is_cloud_ai", False)

    st.markdown("<hr style='margin: 1.5rem 0 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    mode_text = "🌊 DigitalOcean Gradient AI" if is_cloud_ai else "⚡ RepoGuide Smart Analyzer"
    
    st.markdown(f"""
    <div class="white-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <h2 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.65rem; font-weight: 800;">{info.get('full_name')}</h2>
                <p style="margin: 0 0 14px 0; color: #475569; font-size: 0.98rem;">{info.get('description') or 'No description provided.'}</p>
            </div>
            <div style="background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; padding: 5px 14px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">{mode_text}</div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            <span style="background: #F1F5F9; padding: 5px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: #334155;">💻 {info.get('language') or 'General'}</span>
            <span style="background: #F1F5F9; padding: 5px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: #334155;">⭐ {info.get('stargazers_count', 0):,} stars</span>
            <span style="background: #F1F5F9; padding: 5px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: #334155;">🍴 {info.get('forks_count', 0):,} forks</span>
            <span style="background: #F1F5F9; padding: 5px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: #334155;">🌿 {info.get('default_branch', 'main')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_guide, tab_issues, tab_chat, tab_tree, tab_checklist = st.tabs([
        "🧭 Beginner Guide",
        "🎯 Good First Issues & PR",
        "💬 Ask RepoGuide AI",
        "🌳 Architecture & Files",
        "✅ Pre-PR Checklist"
    ])

    with tab_guide:
        st.markdown(guide_content)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Guide as Markdown (.md)",
            data=guide_content,
            file_name=f"{owner}_{repo}_repoguide.md",
            mime="text/markdown",
            key="btn_download_guide_live"
        )

    with tab_issues:
        st.subheader("🎯 Open Good First Issues")
        if issues:
            for i in issues:
                st.markdown(f"""
                <div class="issue-card-clean">
                    <div style="font-size: 1rem; font-weight: 700; margin-bottom: 4px;">
                        <a href="{i['html_url']}" target="_blank" style="color: #10B981; text-decoration: none;">#{i['number']} {i['title']}</a>
                    </div>
                    <div style="color: #64748B; font-size: 0.84rem;">
                        💬 {i.get('comments', 0)} comments &nbsp;•&nbsp; 🟢 State: {i.get('state', 'open')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            st.subheader("🛠️ Step-by-Step PR Plan Generator")
            st.caption("Select an issue below to generate tailored Git commands, files to inspect, and ready-to-copy PR template:")

            issue_options = {f"#{i['number']} - {i['title']}": i for i in issues}
            selected_issue_title = st.selectbox("Choose an issue to work on:", list(issue_options.keys()), key="live_issue_sel")
            selected_issue = issue_options[selected_issue_title]

            if st.button("📝 Generate My First PR Plan", key="btn_pr_plan_live"):
                with st.spinner("Analyzing issue context and drafting PR strategy..."):
                    issue_detail = fetch_issue_body(owner, repo, selected_issue['number'])
                    issue_body = issue_detail.get("body", selected_issue.get("title", ""))[:1500]

                    active_key = secret("DO_MODEL_ACCESS_KEY")
                    if active_key:
                        pr_prompt = f"""Create a beginner-friendly Pull Request step-by-step execution plan.
Repo: {info.get('full_name')}
Language: {info.get('language')}
Target Issue: #{selected_issue['number']} - {selected_issue['title']}
Issue Description:
{issue_body}

Files in repo:
{chr(10).join(paths[:150])}

Explain in simple {lang}:
1. Exact Git commands to fork, clone, create a feature branch (`fix-issue-{selected_issue['number']}`), and install dependencies.
2. Suggested files in the repo to explore based on the issue description.
3. Git commit message following Conventional Commits format (`fix: ...`).
4. Git push command.
5. Ready-to-copy PR title and markdown description template linking to the issue."""
                        try:
                            pr_plan = ask_llm(
                                prompt=pr_prompt,
                                model=secret("DO_MODEL_ID", DEFAULT_MODEL),
                                base_url=secret("DO_BASE_URL", DEFAULT_BASE_URL),
                                api_key=active_key
                            )
                        except Exception:
                            active_key = None

                    if not active_key:
                        pr_plan = f"""### 🚀 Git Workflow Plan for #{selected_issue['number']}

#### 1. Fork & Clone Repository
```bash
# 1. Fork on GitHub, then clone your personal fork:
git clone https://github.com/<your-username>/{repo}.git
cd {repo}

# 2. Create and switch to a dedicated feature branch:
git checkout -b fix-issue-{selected_issue['number']}
```

#### 2. Make Your Fix & Test
- Inspect relevant codebase files related to `{selected_issue['title']}`
- Run project test suites to verify functionality.

#### 3. Commit & Push Changes
```bash
git add .
git commit -m "fix: resolve #{selected_issue['number']} - {selected_issue['title']}"
git push origin fix-issue-{selected_issue['number']}
```

#### 4. Pull Request Title & Description:
- **PR Title:** `fix: resolve #{selected_issue['number']} - {selected_issue['title']}`
- **Description:**
  > Closes #{selected_issue['number']}
  >
  > ### Summary of Changes
  > - Addressed issue regarding {selected_issue['title']}
  > - Tested locally and verified tests pass.
"""
                    st.markdown(pr_plan)
        else:
            st.info("No open 'good first issue' tickets found in this repo right now. Check the main Issues tab on GitHub!")

    with tab_chat:
        st.subheader("💬 Ask RepoGuide AI")
        st.caption(f"Have questions about `{info.get('full_name')}`? Ask about code structure, configuration, running tests, or architecture in {lang}!")

        if "repo_chat_history" not in st.session_state:
            st.session_state["repo_chat_history"] = []

        for msg in st.session_state["repo_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input(f"Ask a question about {info.get('name', 'this repo')}...")
        if user_question:
            st.session_state["repo_chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            active_key = secret("DO_MODEL_ACCESS_KEY")
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if active_key:
                        chat_prompt = f"""You are a helpful open-source mentor answering a beginner's question about this repository.
Repository: {info.get('full_name')}
Description: {info.get('description')}
Main Language: {info.get('language')}

README excerpt:
{readme[:4000]}

Available file paths:
{chr(10).join(paths[:150])}

User question: {user_question}
Answer clearly and warmly in {lang}."""
                        try:
                            ai_reply = ask_llm(chat_prompt, api_key=active_key)
                        except Exception:
                            ai_reply = f"In `{info.get('full_name')}`, refer to `{info.get('language')}` source modules and the README for details."
                    else:
                        ai_reply = f"Based on `{info.get('full_name')}`, the primary codebase is written in **{info.get('language', 'code')}**. You can inspect the main source files in Tab 4 or check setup steps in Tab 1!"

                    st.markdown(ai_reply)
                    st.session_state["repo_chat_history"].append({"role": "assistant", "content": ai_reply})

    with tab_tree:
        st.subheader("🌳 File Tree & Architecture Categorization")
        st.caption(f"Browse key files across {len(paths)} detected source paths.")
        
        categories = classify_file_tree(paths)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            with st.expander(f"🐍 Source & Logic ({len(categories['Source & Logic'])})", expanded=True):
                for p in categories['Source & Logic'][:30]:
                    st.markdown(f"- `{p}`")
                if len(categories['Source & Logic']) > 30:
                    st.caption(f"...and {len(categories['Source & Logic']) - 30} more source files.")

            with st.expander(f"⚙️ Configuration & Build ({len(categories['Configuration & Build'])})", expanded=True):
                for p in categories['Configuration & Build']:
                    st.markdown(f"- `{p}`")

        with col_c2:
            with st.expander(f"🧪 Tests & Quality ({len(categories['Tests & Quality'])})", expanded=True):
                for p in categories['Tests & Quality'][:25]:
                    st.markdown(f"- `{p}`")
                if len(categories['Tests & Quality']) > 25:
                    st.caption(f"...and {len(categories['Tests & Quality']) - 25} more test files.")

            with st.expander(f"📚 Documentation & Community ({len(categories['Documentation & Community'])})", expanded=True):
                for p in categories['Documentation & Community']:
                    st.markdown(f"- `{p}`")

    with tab_checklist:
        st.subheader("✅ Open-Source Contribution Pre-Flight Checklist")
        st.caption("Complete these 6 steps before submitting your Pull Request:")

        c1 = st.checkbox("1. Did you fork the repo and create a new feature branch? (Never commit directly to main)", value=False, key="live_chk_1")
        c2 = st.checkbox("2. Did you pull the latest upstream changes before starting? (`git pull upstream main`)", value=False, key="live_chk_2")
        c3 = st.checkbox("3. Did you follow code formatting and style guidelines? (Check contributing guide)", value=False, key="live_chk_3")
        c4 = st.checkbox("4. Did you run the existing test suite and verify all tests pass?", value=False, key="live_chk_4")
        c5 = st.checkbox("5. Did you write or update tests for your fix/feature?", value=False, key="live_chk_5")
        c6 = st.checkbox("6. Did you link the issue in your PR body (`Closes #<issue_number>`)?", value=False, key="live_chk_6")

        score = sum([c1, c2, c3, c4, c5, c6])
        st.progress(score / 6)
        if score == 6:
            st.success("🎉 You are 100% ready to submit your Pull Request!")
        else:
            st.info(f"Progress: {score}/6 items completed.")

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
        st.subheader("📜 Beginner Git Command Reference")
        st.markdown("""
        | Step | Command | Explanation |
        | :--- | :--- | :--- |
        | **Fork & Clone** | `git clone https://github.com/<your-user>/<repo>.git` | Apna local copy banana |
        | **New Branch** | `git checkout -b fix-my-issue` | Naya isolated branch create karna |
        | **Check Status** | `git status` | Konsi files change hui hain check karna |
        | **Stage Changes** | `git add .` | Modified files ko commit ke liye ready karna |
        | **Commit** | `git commit -m "fix: your clear message"` | Changes ko save karna |
        | **Push** | `git push origin fix-my-issue` | GitHub pe apna branch upload karna |
        """)

# ==========================================
# 3. CORE FEATURES SECTION (#features)
# ==========================================
st.markdown("""
<div class="section-wrapper" id="features">
    <div class="section-tag">Core Capabilities</div>
    <div class="section-title">✨ Everything you need to contribute with confidence</div>
    <div class="section-desc">RepoGuide bridges the gap between intimidating codebases and beginner contributors through open-weight AI models.</div>
</div>
""", unsafe_allow_html=True)

# Multilingual Showcase Card
st.markdown("""
<div class="white-card" style="margin-bottom: 24px;">
    <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">🌐 Multilingual AI Mentorship (Side-by-Side Preview)</div>
    <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 16px;">RepoGuide explains open-source architecture in the language Indian and global developers are most comfortable with.</p>
</div>
""", unsafe_allow_html=True)

c_lang1, c_lang2, c_lang3 = st.columns(3)
with c_lang1:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1.5px solid #10B981; border-radius: 14px; padding: 18px; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08); height: 100%;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #10B981; text-transform: uppercase;">🇮🇳 Hinglish (Default)</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 6px 0;">Beginner Friendly</div>
        <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
            "FastAPI ek high-performance Python framework hai. Agar aapko auth fix karna hai, toh <code>app/routers/auth.py</code> ko check karein aur tests run karne ke liye <code>pytest</code> execute karein."
        </p>
    </div>
    """, unsafe_allow_html=True)

with c_lang2:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px; height: 100%;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #0284C7; text-transform: uppercase;">🇮🇳 Pure Hindi</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 6px 0;">सरल हिंदी में समझें</div>
        <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
            "यह एक ओपन-सोर्स प्रोजेक्ट है। शुरुआत करने के लिए पहले <code>git clone</code> करें, फिर रिपॉजिटरी की मुख्य फाइलों को देखें और पहला पुल रिक्वेस्ट बनाएं।"
        </p>
    </div>
    """, unsafe_allow_html=True)

with c_lang3:
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px; height: 100%;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #8B5CF6; text-transform: uppercase;">🌐 English</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin: 6px 0;">Standard Global</div>
        <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
            "FastAPI is a modern web framework. Core dependencies live in <code>pyproject.toml</code>. Follow the 4-step setup instructions to launch the dev server."
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# 6 Feature Cards
f_c1, f_c2, f_c3 = st.columns(3, gap="medium")
with f_c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">🎯</div>
        <div class="feature-title">Live Issue Triage</div>
        <div class="feature-text">Automatically pulls open, verified <code>good first issue</code> tickets via GitHub REST API, ignoring PRs so you only tackle genuine beginner tasks.</div>
    </div>
    """, unsafe_allow_html=True)
with f_c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">🛠️</div>
        <div class="feature-title">1-Click PR Plan Generator</div>
        <div class="feature-text">Select any issue to generate step-by-step Git bash commands, suggested files to inspect, Conventional Commits, and ready-to-paste PR descriptions.</div>
    </div>
    """, unsafe_allow_html=True)
with f_c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">🌳</div>
        <div class="feature-title">Recursive Code Mapping</div>
        <div class="feature-text">Recursively analyzes the git tree up to 250 files, organizing files into Source, Config, Tests, and Documentation. Zero hallucinations.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

f_c4, f_c5, f_c6 = st.columns(3, gap="medium")
with f_c4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">💬</div>
        <div class="feature-title">Ask RepoGuide AI Chatbot</div>
        <div class="feature-text">Have follow-up questions? Ask the interactive chatbot about test setups, database models, or API routing grounded in the codebase.</div>
    </div>
    """, unsafe_allow_html=True)
with f_c5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">⚡</div>
        <div class="feature-title">DigitalOcean Gradient AI</div>
        <div class="feature-text">Powered by serverless open-weight models (Alibaba Qwen3-32B, Llama-3.1-70B) served with low-latency and auto-sanitized reasoning tags.</div>
    </div>
    """, unsafe_allow_html=True)
with f_c6:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon-box">✅</div>
        <div class="feature-title">Pre-Flight PR Checklist</div>
        <div class="feature-text">An interactive 6-step checklist with a live progress bar ensuring your Pull Request adheres to open-source guidelines before you submit.</div>
    </div>
    """, unsafe_allow_html=True)

# Comparison Table
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="white-card">
    <h3 style="margin-top: 0; color: #0F172A;">📊 Why RepoGuide beats generic AI & manual browsing:</h3>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem; color: #334155;">
        <thead>
            <tr style="border-bottom: 2px solid #E2E8F0; text-align: left;">
                <th style="padding: 12px 10px;">Capability</th>
                <th style="padding: 12px 10px; color: #10B981;">RepoGuide (DigitalOcean AI)</th>
                <th style="padding: 12px 10px; color: #64748B;">Standard GitHub</th>
                <th style="padding: 12px 10px; color: #64748B;">Generic ChatGPT</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 12px 10px; font-weight: 600;">Real File Tree Grounding</td>
                <td style="padding: 12px 10px; color: #10B981; font-weight: 700;">✅ Real 250 paths injected</td>
                <td style="padding: 12px 10px;">⚠️ Raw manual browsing</td>
                <td style="padding: 12px 10px;">❌ Hallucinates fake files</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 12px 10px; font-weight: 600;">Live Good-First-Issues</td>
                <td style="padding: 12px 10px; color: #10B981; font-weight: 700;">✅ Live GitHub API extraction</td>
                <td style="padding: 12px 10px;">⚠️ Manual filter search</td>
                <td style="padding: 12px 10px;">❌ Cannot fetch live tickets</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 12px 10px; font-weight: 600;">Multilingual Explanations</td>
                <td style="padding: 12px 10px; color: #10B981; font-weight: 700;">✅ Hinglish, Hindi, English</td>
                <td style="padding: 12px 10px;">❌ English only</td>
                <td style="padding: 12px 10px;">⚠️ Requires complex prompting</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 12px 10px; font-weight: 600;">Automated PR Plan Generator</td>
                <td style="padding: 12px 10px; color: #10B981; font-weight: 700;">✅ Exact Git bash commands</td>
                <td style="padding: 12px 10px;">❌ None</td>
                <td style="padding: 12px 10px;">⚠️ Generic advice</td>
            </tr>
            <tr>
                <td style="padding: 12px 10px; font-weight: 600;">Open-Weight AI Model</td>
                <td style="padding: 12px 10px; color: #10B981; font-weight: 700;">✅ Qwen3-32B / Llama-3.1</td>
                <td style="padding: 12px 10px;">❌ None</td>
                <td style="padding: 12px 10px;">❌ Proprietary Closed API</td>
            </tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. HOW IT WORKS SECTION (#how-it-works)
# ==========================================
st.markdown("""
<div class="section-wrapper" id="how-it-works">
    <div class="section-tag">Workflow</div>
    <div class="section-title">⚙️ From Codebase Confusion to Merged PR in 4 Steps</div>
    <div class="section-desc">RepoGuide eliminates friction by automating repo context extraction and generating a clear roadmap.</div>
</div>
""", unsafe_allow_html=True)

s_col1, s_col2, s_col3, s_col4 = st.columns(4, gap="medium")

with s_col1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">01</div>
        <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; color: #0F172A;">Paste Repository</div>
        <div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">Enter any public GitHub URL (e.g. FastAPI or your personal project) and choose your preferred language.</div>
    </div>
    """, unsafe_allow_html=True)

with s_col2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">02</div>
        <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; color: #0F172A;">GitHub Extraction</div>
        <div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">GitHub REST API extracts the full README, recursive file tree (up to 250 paths), and open beginner issues.</div>
    </div>
    """, unsafe_allow_html=True)

with s_col3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">03</div>
        <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; color: #0F172A;">Gradient AI Inference</div>
        <div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">Open-source LLM on DigitalOcean analyzes the repo structure, stripping hidden think tags and preventing hallucinations.</div>
    </div>
    """, unsafe_allow_html=True)

with s_col4:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">04</div>
        <div style="font-weight: 800; font-size: 1.05rem; margin-bottom: 6px; color: #0F172A;">Execute & Contribute</div>
        <div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">Get an onboarding map, interactive Q&A, and exact git commands to create your branch and submit your PR!</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. ARCHITECTURE & DIGITALOCEAN AI (#architecture)
# ==========================================
st.markdown("""
<div class="section-wrapper" id="architecture">
    <div class="section-tag">Infrastructure</div>
    <div class="section-title">Built with Open-Weight AI on DigitalOcean</div>
    <div class="section-desc">Designed to compete for the <strong>Best Open-Source AI Project (DigitalOcean)</strong> prize at Hacktoberfest Hack Day Delhi.</div>
</div>
""", unsafe_allow_html=True)

arch_col1, arch_col2 = st.columns([1.1, 0.9], gap="large")

with arch_col1:
    st.markdown("""
    <div class="white-card">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.25rem;">Why Open-Weight AI on DigitalOcean?</h3>
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
            RepoGuide avoids proprietary closed-source models. It leverages <strong>DigitalOcean Serverless Inference (Gradient AI)</strong> with models like <strong>Alibaba Qwen3-32B</strong> and <strong>Meta Llama-3.1-70B</strong>.
        </p>
        <div style="margin-top: 16px; font-size: 0.9rem; line-height: 1.8; color: #334155;">
            <div><strong style="color: #10B981;">• Base URL:</strong> <code>https://inference.do-ai.run/v1</code></div>
            <div><strong style="color: #10B981;">• Open Models:</strong> <code>alibaba-qwen3-32b</code>, <code>llama-3.1-70b-instruct</code></div>
            <div><strong style="color: #10B981;">• SDK Protocol:</strong> Standard <code>openai</code> Python client format</div>
            <div><strong style="color: #10B981;">• Deployment:</strong> DigitalOcean App Platform (Port 8080)</div>
            <div><strong style="color: #10B981;">• Fallback Engine:</strong> Zero-downtime offline Smart Analyzer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with arch_col2:
    st.markdown("""
    <div class="white-card">
        <h3 style="margin-top: 0; color: #0F172A; font-size: 1.25rem;">Project Environment & Security</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #334155;">
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 10px 0; font-weight: 700;">Frontend</td>
                <td style="padding: 10px 0;">Streamlit + Plus Jakarta Sans</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 10px 0; font-weight: 700;">Secrets Protection</td>
                <td style="padding: 10px 0;"><code>.env</code> & <code>.streamlit/secrets.toml</code> git-ignored</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 10px 0; font-weight: 700;">License</td>
                <td style="padding: 10px 0;">MIT Open-Source License</td>
            </tr>
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 10px 0; font-weight: 700;">Skill Integration</td>
                <td style="padding: 10px 0;"><code>SKILL.md</code> agent workflow included</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: 700;">Author</td>
                <td style="padding: 10px 0;">Yash Kumar (DTU)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. GLOBAL FOOTER
# ==========================================
st.markdown("""
<div style="margin-top: 60px; padding: 26px 0; border-top: 1px solid #E2E8F0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; color: #64748B; font-size: 0.88rem;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-weight: 800; color: #0F172A; font-size: 1.05rem;">🧭 RepoGuide</span>
        <span>•</span>
        <span>Author: Yash Kumar (<a href="http://dtu.ac.in/" target="_blank" style="color: #10B981; text-decoration: none;">Delhi Technological University</a>)</span>
    </div>
    <div>
        <span>Hacktoberfest Hack Day Delhi x Braveon AI & RoundTable DTU</span>
        <span>•</span>
        <a href="https://github.com/coder-Yash886/OpenPath" target="_blank" style="color: #10B981; text-decoration: none; font-weight: 600;">GitHub Repository</a>
    </div>
</div>
""", unsafe_allow_html=True)
