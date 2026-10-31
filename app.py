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
    "https://github.com/coder-Yash886/Nexora-AI",
    "https://github.com/fastapi/fastapi",
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests"
]

# ==========================================
# Helper Functions
# ==========================================
def secret(name, default=""):
    """Fetch secret from Streamlit secrets, environment, or session state."""
    if f"ui_{name}" in st.session_state and st.session_state[f"ui_{name}"]:
        return st.session_state[f"ui_{name}"]
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

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
        raise RuntimeError("Repository not found, or it is private. Please check the URL.")
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
    
    # Fetch README
    readme = ""
    try:
        data = gh_get(f"/repos/{owner}/{repo}/readme")
        readme = base64.b64decode(data["content"]).decode("utf-8", "ignore")
    except Exception:
        pass

    # Fetch recursive file tree
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

    # Fetch good first issues
    issues = []
    try:
        issues_raw = gh_get(
            f"/repos/{owner}/{repo}/issues",
            {"labels": "good first issue", "state": "open", "per_page": 6},
        )
        issues = [i for i in issues_raw if "pull_request" not in i]
    except Exception:
        pass

    # Fallback to general open issues if no 'good first issue' label exists
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
    desc = info.get('description') or f"Open-source software project built with {info.get('language', 'modern tools')}."
    lang_name = info.get('language') or 'General'
    
    # 1. What this project does
    if lang == "Hinglish":
        intro = f"**{repo_name}** ek open-source project hai jo {desc.lower() if desc else 'developers ke kaam aati hai'}.\nYeh project primary taur par **{lang_name}** mein likha gaya hai aur iska goal developers aur users ko fast aur modular solution provide karna hai."
    elif lang == "Hindi":
        intro = f"**{repo_name}** एक ओपन-सोर्स प्रोजेक्ट है। {desc}\nयह मुख्य रूप से **{lang_name}** भाषा में लिखा गया है और इसका उद्देश्य डेवलपर्स को एक सरल और उपयोगी टूल प्रदान करना है।"
    else:
        intro = f"**{repo_name}** is an open-source project designed to provide {desc.lower() if desc else 'powerful utilities'}.\nIt is primarily written in **{lang_name}** and designed to be modular, easy to understand, and developer-friendly."

    # 2. Where things live
    tree_categories = classify_file_tree(paths)
    key_files = []
    if tree_categories["Source & Logic"]:
        key_files.append(f"- **`{tree_categories['Source & Logic'][0]}`**: Core application logic & primary implementation")
        if len(tree_categories["Source & Logic"]) > 1:
            key_files.append(f"- **`{tree_categories['Source & Logic'][1]}`**: Module utilities & helper functions")
    if tree_categories["Configuration & Build"]:
        key_files.append(f"- **`{tree_categories['Configuration & Build'][0]}`**: Build configuration and project dependencies")
    if tree_categories["Tests & Quality"]:
        key_files.append(f"- **`{tree_categories['Tests & Quality'][0]}`**: Automated test suite for validating changes")
    if tree_categories["Documentation & Community"]:
        key_files.append(f"- **`{tree_categories['Documentation & Community'][0]}`**: Project documentation and contributor guidelines")
    
    if not key_files:
        key_files = [f"- `{p}`: Repository root file" for p in paths[:5]]
    
    map_text = "\n".join(key_files)

    # 3. How to run it locally
    run_steps = []
    run_steps.append(f"1. **Clone the repository:**\n   ```bash\n   git clone https://github.com/{full_name}.git\n   cd {repo_name}\n   ```")
    
    if "python" in lang_name.lower() or any("requirements" in p.lower() or "pyproject" in p.lower() for p in paths):
        run_steps.append("2. **Create virtual environment & install requirements:**\n   ```bash\n   python -m venv venv\n   source venv/bin/activate  # Windows: venv\\Scripts\\activate\n   pip install -r requirements.txt\n   ```")
        run_steps.append("3. **Run the project:**\n   Check the main script (e.g. `python app.py` or `python main.py`).")
    elif "javascript" in lang_name.lower() or "typescript" in lang_name.lower() or any("package.json" in p.lower() for p in paths):
        run_steps.append("2. **Install Node dependencies:**\n   ```bash\n   npm install\n   ```")
        run_steps.append("3. **Start local development server:**\n   ```bash\n   npm run dev  # or npm start\n   ```")
    else:
        run_steps.append("2. **Follow README instructions:**\n   Review the `README.md` for specific language dependencies and compile steps.")
    
    local_text = "\n".join(run_steps)

    # 4. Where to start contributing
    contrib_points = []
    if issues:
        first_issue = issues[0]
        contrib_points.append(f"1. **Solve open Issue #{first_issue['number']}**: [{first_issue['title']}]({first_issue['html_url']})")
    else:
        contrib_points.append("1. **Documentation improvements**: Look for missing setup instructions or typos in docs.")
    contrib_points.append("2. **Add or enhance test coverage**: Add a test case for edge scenarios in the tests directory.")
    contrib_points.append("3. **Open a clean Pull Request**: Create a new branch `fix/my-feature` and link your PR to the issue!")
    
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
    page_title="RepoGuide | AI Open-Source Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Clean White / Light Theme CSS
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    
    /* Header & Titles */
    .hero-container {
        padding: 1.5rem 0 1rem 0;
        text-align: left;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0069FF;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #EFF6FF;
        color: #0069FF;
        border: 1px solid #BFDBFE;
        margin-bottom: 0.8rem;
    }
    .sub-caption {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    }
    .metric-badge {
        display: inline-block;
        background: #F1F5F9;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #334155;
        font-weight: 500;
        margin-right: 8px;
    }

    /* Issue Card */
    .issue-card-white {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
    }
    .issue-card-white:hover {
        border-color: #0069FF;
        box-shadow: 0 4px 12px rgba(0, 105, 255, 0.1);
        transform: translateY(-1px);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .sample-chip {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        color: #1E293B !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
    }
    
    /* Code blocks */
    code {
        background-color: #F1F5F9 !important;
        color: #0069FF !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.88em !important;
    }
    pre {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Hero Header
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="badge-pill">🌊 DigitalOcean Gradient AI • Open-Source Companion</div>
    <div class="main-title">🧭 RepoGuide</div>
    <div class="sub-caption">Paste any public GitHub repository. Get an instant beginner-friendly guide, architecture breakdown, interactive Q&A, and PR plan in Hinglish, Hindi, or English.</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# Collapsible Settings Bar (Directly in Main View)
# ==========================================
with st.expander("⚙️ **DigitalOcean AI & Cloud Settings** (Click to configure API Key or switch models)", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        ui_do_key = st.text_input(
            "🔑 DigitalOcean Model Access Key",
            value=secret("DO_MODEL_ACCESS_KEY"),
            type="password",
            help="Your Model Access Key from DigitalOcean Inference Hub / Gradient AI.",
            key="ui_DO_MODEL_ACCESS_KEY"
        )
        model_choice = st.selectbox(
            "🤖 Open-Source LLM Model",
            options=SUPPORTED_MODELS,
            index=0,
            help="Select open-weight models available on DigitalOcean Gradient AI."
        )
        if model_choice == "Custom...":
            active_model = st.text_input("Custom Model ID", value="alibaba-qwen3-32b", key="ui_DO_MODEL_ID")
        else:
            active_model = model_choice
            st.session_state["ui_DO_MODEL_ID"] = active_model

    with col_s2:
        ui_base_url = st.text_input(
            "🌐 Inference Base URL",
            value=secret("DO_BASE_URL", DEFAULT_BASE_URL),
            help="DigitalOcean serverless inference base URL.",
            key="ui_DO_BASE_URL"
        )
        ui_gh_token = st.text_input(
            "🐙 GitHub Token (Optional)",
            value=secret("GITHUB_TOKEN"),
            type="password",
            help="Personal Access Token avoids unauthenticated 60 req/hr rate limits.",
            key="ui_GITHUB_TOKEN"
        )
    st.caption("💡 *Note: If no DigitalOcean key is entered, RepoGuide runs in **Smart Local Analysis Mode** so you can test seamlessly!*")

# ==========================================
# Quick Sample Repository Chips
# ==========================================
st.markdown("<p style='font-weight: 600; color: #475569; margin-bottom: 6px;'>⚡ Quick test with popular repositories:</p>", unsafe_allow_html=True)
cols = st.columns(len(SAMPLE_REPOS))
for idx, s_url in enumerate(SAMPLE_REPOS):
    repo_name = s_url.split("/")[-1]
    if cols[idx].button(f"📦 {repo_name}", key=f"btn_sample_{idx}", use_container_width=True):
        st.session_state["target_repo_url"] = s_url
        st.rerun()

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==========================================
# Main Input Section
# ==========================================
default_input_url = st.session_state.get("target_repo_url", "")
col_url, col_lang = st.columns([3, 1])

with col_url:
    url = st.text_input(
        "GitHub repository URL",
        value=default_input_url,
        placeholder="https://github.com/owner/repository",
        label_visibility="collapsed"
    )

with col_lang:
    lang = st.selectbox("Language", ["Hinglish", "Hindi", "English"], label_visibility="collapsed")

col_btn, _ = st.columns([1, 3])
with col_btn:
    submit_button = st.button("🚀 Explain this repo", type="primary", use_container_width=True)

# ==========================================
# Execution Engine
# ==========================================
if submit_button and url:
    try:
        owner, repo = parse_repo(url)
        
        with st.spinner(f"🔍 Reading repository '{owner}/{repo}' via GitHub API..."):
            info, readme, paths, issues = fetch_repo(owner, repo)

        st.session_state["current_info"] = info
        st.session_state["current_readme"] = readme
        st.session_state["current_paths"] = paths
        st.session_state["current_issues"] = issues
        st.session_state["current_owner"] = owner
        st.session_state["current_repo"] = repo
        st.session_state["current_lang"] = lang
        st.session_state["repo_chat_history"] = []

        # Check if DigitalOcean Key is available
        active_key = secret("DO_MODEL_ACCESS_KEY")
        base_url_to_use = secret("DO_BASE_URL", DEFAULT_BASE_URL)
        model_to_use = secret("DO_MODEL_ID", DEFAULT_MODEL)

        if active_key:
            with st.spinner(f"🧠 Calling DigitalOcean Gradient AI ({model_to_use})..."):
                issue_text = "\n".join(
                    f"#{i['number']} {i['title']} ({i['html_url']})" for i in issues
                ) or "No open 'good first issue' tickets found."

                prompt = f"""You are a friendly open-source mentor for beginners.
Explain this GitHub repository in simple {lang}.

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
                    st.warning(f"⚠️ DigitalOcean Cloud Inference notice: {llm_err}. Using Smart Local Analyzer fallback.")
                    answer = generate_smart_guide(info, readme, paths, issues, lang)
                    st.session_state["is_cloud_ai"] = False
        else:
            with st.spinner("⚡ Analyzing codebase via RepoGuide Smart Engine..."):
                answer = generate_smart_guide(info, readme, paths, issues, lang)
                st.session_state["is_cloud_ai"] = False

        st.session_state["current_guide"] = answer

    except ValueError as ve:
        st.warning(f"⚠️ {ve}")
    except RuntimeError as re_err:
        st.error(f"❌ {re_err}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

# ==========================================
# Display Tabs Interface
# ==========================================
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

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # Repository Overview Banner (Clean White Card)
    mode_pill = "🌊 DigitalOcean Gradient Cloud AI" if is_cloud_ai else "⚡ RepoGuide Smart Analyzer"
    
    st.markdown(f"""
    <div class="white-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h2 style="margin: 0 0 6px 0; color: #0069FF; font-size: 1.6rem; font-weight: 700;">{info.get('full_name')}</h2>
                <p style="margin: 0 0 12px 0; color: #475569; font-size: 0.95rem;">{info.get('description') or 'No description provided.'}</p>
            </div>
            <span class="badge-pill">{mode_pill}</span>
        </div>
        <div>
            <span class="metric-badge">💻 <strong>Language:</strong> {info.get('language') or 'N/A'}</span>
            <span class="metric-badge">⭐ <strong>Stars:</strong> {info.get('stargazers_count', 0):,}</span>
            <span class="metric-badge">🍴 <strong>Forks:</strong> {info.get('forks_count', 0):,}</span>
            <span class="metric-badge">🌿 <strong>Default Branch:</strong> <code>{info.get('default_branch', 'main')}</code></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Navigation Tabs
    tab_guide, tab_issues, tab_chat, tab_tree, tab_checklist = st.tabs([
        "🧭 Beginner Guide",
        "🎯 Good First Issues & PR Plan",
        "💬 Ask RepoGuide AI",
        "🌳 File Tree & Architecture",
        "✅ Pre-PR Checklist & Git Guide"
    ])

    # ------------------------------------------
    # TAB 1: Beginner Guide
    # ------------------------------------------
    with tab_guide:
        st.markdown(guide_content)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Guide as Markdown (.md)",
            data=guide_content,
            file_name=f"{owner}_{repo}_repoguide.md",
            mime="text/markdown"
        )

    # ------------------------------------------
    # TAB 2: Good First Issues & PR Plan Generator
    # ------------------------------------------
    with tab_issues:
        st.subheader("🎯 Open Good First Issues")
        if issues:
            for i in issues:
                st.markdown(f"""
                <div class="issue-card-white">
                    <div style="font-size: 1rem; font-weight: 600; margin-bottom: 4px;">
                        <a href="{i['html_url']}" target="_blank" style="color: #0069FF; text-decoration: none;">#{i['number']} {i['title']}</a>
                    </div>
                    <div style="color: #64748B; font-size: 0.82rem;">
                        💬 Comments: {i.get('comments', 0)} &nbsp;|&nbsp; 🟢 State: {i.get('state', 'open')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
            st.subheader("🛠️ Step-by-Step PR Plan Generator")
            st.caption("Select an issue below to generate tailored Git bash commands, files to inspect, and PR template!")

            issue_options = {f"#{i['number']} - {i['title']}": i for i in issues}
            selected_issue_title = st.selectbox("Choose issue to work on:", list(issue_options.keys()))
            selected_issue = issue_options[selected_issue_title]

            if st.button("📝 Generate My First PR Plan", key="btn_pr_plan"):
                with st.spinner("Fetching full issue context and drafting PR strategy..."):
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
# 1. Fork the repo on GitHub, then clone your fork:
git clone https://github.com/<your-username>/{repo}.git
cd {repo}

# 2. Create and switch to a dedicated feature branch:
git checkout -b fix-issue-{selected_issue['number']}
```

#### 2. Make Your Fix & Test
- Inspect relevant codebase files related to: `{selected_issue['title']}`
- Run local unit tests to ensure existing tests pass.

#### 3. Commit & Push
```bash
git add .
git commit -m "fix: resolve #{selected_issue['number']} - {selected_issue['title']}"
git push origin fix-issue-{selected_issue['number']}
```

#### 4. Ready-to-Copy Pull Request Template:
- **PR Title:** `fix: resolve #{selected_issue['number']} - {selected_issue['title']}`
- **Description:**
  > Closes #{selected_issue['number']}
  >
  > ### Summary of Changes
  > - Resolved issue regarding {selected_issue['title']}
  > - Verified functionality and ran local tests.
"""
                    st.markdown(pr_plan)
        else:
            st.info("No open 'good first issue' tickets found in this repo right now. Check the main Issues tab on GitHub!")

    # ------------------------------------------
    # TAB 3: Interactive Q&A (Ask RepoGuide AI)
    # ------------------------------------------
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
                            ai_reply = f"In `{info.get('full_name')}`, you can check `{info.get('language')}` files and the README for setup and usage instructions."
                    else:
                        ai_reply = f"Based on `{info.get('full_name')}`, the primary codebase is written in **{info.get('language', 'code')}**. You can inspect the main source files in the file tree or run the local tests mentioned in Tab 1!"

                    st.markdown(ai_reply)
                    st.session_state["repo_chat_history"].append({"role": "assistant", "content": ai_reply})

    # ------------------------------------------
    # TAB 4: File Tree & Architecture Explorer
    # ------------------------------------------
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

    # ------------------------------------------
    # TAB 5: Pre-PR Checklist & Git Guide
    # ------------------------------------------
    with tab_checklist:
        st.subheader("✅ Open-Source Contribution Pre-Flight Checklist")
        st.caption("Check these off before opening your Pull Request to ensure maintainers love your contribution!")

        c1 = st.checkbox("1. Did you fork the repo and create a new feature branch? (Never commit directly to main)", value=False)
        c2 = st.checkbox("2. Did you pull the latest upstream changes before starting? (`git pull upstream main`)", value=False)
        c3 = st.checkbox("3. Did you follow code formatting and style guidelines? (Check contributing guide)", value=False)
        c4 = st.checkbox("4. Did you run the existing test suite and verify all tests pass?", value=False)
        c5 = st.checkbox("5. Did you write or update tests for your fix/feature?", value=False)
        c6 = st.checkbox("6. Did you link the issue in your PR body (`Closes #<issue_number>`)?", value=False)

        score = sum([c1, c2, c3, c4, c5, c6])
        st.progress(score / 6)
        if score == 6:
            st.success("🎉 You are 100% ready to submit your Pull Request!")
        else:
            st.info(f"Progress: {score}/6 items completed.")

        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
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
