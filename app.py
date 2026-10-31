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
    "https://github.com/pallets/flask",
    "https://github.com/fastapi/fastapi",
    "https://github.com/psf/requests",
    "https://github.com/tiangolo/typer"
]

# ==========================================
# Helper Functions
# ==========================================
def secret(name, default=""):
    """Fetch secret from Streamlit secrets, environment, or session state."""
    # Check session state override first (from UI settings)
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
        raise RuntimeError("Repo not found, or it is private.")
    if r.status_code == 403:
        raise RuntimeError("GitHub rate limit reached. Add a GITHUB_TOKEN in settings or .streamlit/secrets.toml.")
    r.raise_for_status()
    return r.json()

def parse_repo(url):
    """Extract owner and repository name from GitHub URL."""
    url = url.strip()
    m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        raise ValueError("Please paste a valid GitHub repo URL (e.g., https://github.com/owner/repo).")
    return m.group(1), m.group(2).removesuffix(".git")

def fetch_repo(owner, repo):
    """Fetch repo info, README, recursive tree, and beginner issues."""
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
            {"labels": "good first issue", "state": "open", "per_page": 5},
        )
        # Exclude Pull Requests
        issues = [i for i in issues_raw if "pull_request" not in i]
    except Exception:
        pass

    # Fallback to general open issues if no 'good first issue' label exists
    if not issues:
        try:
            issues_raw = gh_get(
                f"/repos/{owner}/{repo}/issues",
                {"state": "open", "per_page": 5},
            )
            issues = [i for i in issues_raw if "pull_request" not in i]
        except Exception:
            pass

    return info, readme, paths, issues

def ask_llm(prompt, model=None, base_url=None, api_key=None):
    """Query open-weight LLM on DigitalOcean Gradient Serverless Inference."""
    key = api_key or secret("DO_MODEL_ACCESS_KEY")
    url = base_url or secret("DO_BASE_URL", DEFAULT_BASE_URL)
    mod = model or secret("DO_MODEL_ID", DEFAULT_MODEL)

    if not key:
        raise RuntimeError(
            "DigitalOcean Model Access Key is missing! "
            "Please provide DO_MODEL_ACCESS_KEY in the sidebar or in .streamlit/secrets.toml."
        )

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
        # Strip any hidden reasoning text some models add (e.g. DeepSeek/Qwen think tags)
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

def generate_mock_guide(info, paths, issues, lang):
    """Offline demo mode fallback for evaluating without active DigitalOcean API keys."""
    lang_greeting = {
        "Hinglish": "Yeh project ekdam shaandaar open-source initiative hai!",
        "Hindi": "यह प्रोजेक्ट एक बेहतरीन ओपन-सोर्स प्रोजेक्ट है।",
        "English": "This project is a powerful open-source tool designed for developers."
    }.get(lang, "This project is a fantastic open-source project.")

    top_paths = paths[:6] if paths else ["src/main.py", "README.md", "tests/"]
    paths_desc = "\n".join([f"- `{p}`: Core codebase module" for p in top_paths])

    issue_mention = ""
    if issues:
        first_issue = issues[0]
        issue_mention = f"Start with Issue #{first_issue['number']}: **{first_issue['title']}**."
    else:
        issue_mention = "Start by reading the documentation and opening a beginner fix or test."

    return f"""## What this project does
{lang_greeting}
- **Repository**: {info.get('full_name')}
- **Description**: {info.get('description', 'No description provided')}
- **Main Language**: {info.get('language', 'Unknown')}
- **Stars & Forks**: ⭐ {info.get('stargazers_count', 0)} stars | 🍴 {info.get('forks_count', 0)} forks

## Where things live
A bird's-eye map of the key directories and files:
{paths_desc}

## How to run it locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/{info.get('full_name')}.git
   cd {info.get('name', 'project')}
   ```
2. **Install dependencies:**
   Follow setup steps listed in the repository README.
3. **Run tests or the application:**
   Check the test suite or launch command specified in the project files.

## Where to start contributing
1. {issue_mention}
2. Look at small doc typos, missing test coverage, or edge cases.
3. Fork the repo, make your changes on a new branch, and submit a PR!
*(Note: Generated via RepoGuide Offline Demo Mode)*"""

# ==========================================
# Streamlit UI Setup
# ==========================================
st.set_page_config(
    page_title="RepoGuide | AI Open-Source Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Hackathon Aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0069FF 0%, #00D2FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(0, 105, 255, 0.15);
        color: #00D2FF;
        border: 1px solid rgba(0, 210, 255, 0.3);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(22, 30, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 15px;
    }
    .issue-card {
        background: rgba(16, 24, 39, 0.85);
        border: 1px solid rgba(0, 105, 255, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .issue-card:hover {
        border-color: #00D2FF;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar Settings
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ DigitalOcean AI Settings")
    st.caption("Powered by Open-Weight LLMs via DigitalOcean Serverless Inference")
    
    # Model Access Key
    env_do_key = secret("DO_MODEL_ACCESS_KEY")
    ui_do_key = st.text_input(
        "DigitalOcean Model Access Key",
        value=env_do_key,
        type="password",
        help="Model Access Key from DigitalOcean Inference Hub / Gradient AI.",
        key="ui_DO_MODEL_ACCESS_KEY"
    )

    # Model Selection (Extra Feature: Swappable Models)
    model_choice = st.selectbox(
        "Open-Source LLM Model",
        options=SUPPORTED_MODELS,
        index=0,
        help="Select any open-weight model deployed on DigitalOcean serverless inference."
    )
    if model_choice == "Custom...":
        active_model = st.text_input("Custom DO Model ID", value="alibaba-qwen3-32b", key="ui_DO_MODEL_ID")
    else:
        active_model = model_choice
        st.session_state["ui_DO_MODEL_ID"] = active_model

    # Base URL
    env_base_url = secret("DO_BASE_URL", DEFAULT_BASE_URL)
    ui_base_url = st.text_input(
        "Inference Base URL",
        value=env_base_url,
        help="DigitalOcean serverless inference base URL.",
        key="ui_DO_BASE_URL"
    )

    # GitHub Token (Optional)
    env_gh_token = secret("GITHUB_TOKEN")
    ui_gh_token = st.text_input(
        "GitHub Token (Optional)",
        value=env_gh_token,
        type="password",
        help="Personal access token lifts GitHub's rate limit from 60 to 5000 requests/hour.",
        key="ui_GITHUB_TOKEN"
    )

    # Demo Mode Toggle
    st.markdown("---")
    demo_mode = st.toggle("🧪 Demo Mode (Offline Fallback)", value=False, help="Simulate AI responses without requiring active DigitalOcean credits.")

    st.markdown("---")
    st.markdown("""
    **Target Prize:** Best Open-Source AI Project  
    **Hackathon:** Hacktoberfest Hack Day Delhi x Braveon AI & RoundTable DTU  
    **Author:** Yash Kumar (DTU)
    """)

# ==========================================
# Main Interface
# ==========================================
st.markdown('<div class="main-title">🧭 RepoGuide</div>', unsafe_allow_html=True)
st.markdown('<div class="badge">DigitalOcean Gradient AI • Open-Source Companion</div>', unsafe_allow_html=True)
st.caption("Paste any public GitHub repository. Get an instant beginner-friendly guide and contribution map in Hinglish, Hindi, or English.")

# Quick sample repository buttons
st.markdown("**Quick test with popular repositories:**")
cols = st.columns(len(SAMPLE_REPOS))
for idx, s_url in enumerate(SAMPLE_REPOS):
    repo_name = s_url.split("/")[-1]
    if cols[idx].button(f"📦 {repo_name}", key=f"btn_sample_{idx}"):
        st.session_state["target_repo_url"] = s_url
        st.rerun()

# Input URL & Language
default_input_url = st.session_state.get("target_repo_url", "")
col_url, col_lang = st.columns([3, 1])

with col_url:
    url = st.text_input(
        "GitHub repo URL",
        value=default_input_url,
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed"
    )

with col_lang:
    lang = st.selectbox("Language", ["Hinglish", "Hindi", "English"], label_visibility="collapsed")

# Explain Button
col_btn, _ = st.columns([1, 4])
with col_btn:
    submit_button = st.button("🚀 Explain this repo", type="primary", use_container_width=True)

# Main Execution Flow
if submit_button and url:
    try:
        owner, repo = parse_repo(url)
        
        with st.spinner(f"🔍 Reading repository '{owner}/{repo}' via GitHub API..."):
            info, readme, paths, issues = fetch_repo(owner, repo)

        # Store in session state for PR plan generator and persistence
        st.session_state["current_info"] = info
        st.session_state["current_paths"] = paths
        st.session_state["current_issues"] = issues
        st.session_state["current_owner"] = owner
        st.session_state["current_repo"] = repo

        # Display Repo Overview Metrics
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #00D2FF;">{info.get('full_name')}</h3>
            <p style="margin: 5px 0 10px 0; color: #9CA3AF;">{info.get('description') or 'No description provided.'}</p>
            <span>💻 <strong>Language:</strong> {info.get('language') or 'N/A'}</span> &nbsp;|&nbsp;
            <span>⭐ <strong>Stars:</strong> {info.get('stargazers_count', 0):,}</span> &nbsp;|&nbsp;
            <span>🍴 <strong>Forks:</strong> {info.get('forks_count', 0):,}</span> &nbsp;|&nbsp;
            <span>🌿 <strong>Default Branch:</strong> <code>{info.get('default_branch', 'main')}</code></span>
        </div>
        """, unsafe_allow_html=True)

        # Build Prompt for DigitalOcean Serverless Inference
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

        with st.spinner(f"🧠 Generating beginner guide using {active_model} on DigitalOcean..."):
            if demo_mode:
                answer = generate_mock_guide(info, paths, issues, lang)
            else:
                answer = ask_llm(
                    prompt=prompt,
                    model=active_model,
                    base_url=ui_base_url,
                    api_key=ui_do_key
                )

        st.session_state["current_guide"] = answer

    except ValueError as ve:
        st.warning(f"⚠️ {ve}")
    except RuntimeError as re_err:
        st.error(f"❌ {re_err}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

# ==========================================
# Display Generated Guide & Actions
# ==========================================
if "current_guide" in st.session_state and st.session_state["current_guide"]:
    guide_content = st.session_state["current_guide"]
    issues = st.session_state.get("current_issues", [])
    owner = st.session_state.get("current_owner", "")
    repo = st.session_state.get("current_repo", "")
    info = st.session_state.get("current_info", {})

    st.markdown("---")
    st.markdown(guide_content)

    # Download / Export Button (Extra Feature)
    st.download_button(
        label="📥 Download Guide as Markdown (.md)",
        data=guide_content,
        file_name=f"{owner}_{repo}_repoguide.md",
        mime="text/markdown"
    )

    # Clickable Open Issues Section
    st.markdown("---")
    st.subheader("🎯 Open Good First Issues")
    if issues:
        for i in issues:
            st.markdown(f"""
            <div class="issue-card">
                <strong><a href="{i['html_url']}" target="_blank" style="color: #00D2FF; text-decoration: none;">#{i['number']} {i['title']}</a></strong><br>
                <small style="color: #9CA3AF;">Comments: {i.get('comments', 0)} | State: {i.get('state', 'open')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open 'good first issue' tickets found in this repo right now. Check the main Issues tab on GitHub!")

    # ==========================================
    # Optional Extra: Generate My First PR Plan
    # ==========================================
    st.markdown("---")
    st.subheader("🛠️ Step-by-Step PR Plan Generator")
    st.caption("Pick an issue or task to generate step-by-step Git bash commands and a Pull Request template!")

    if issues:
        issue_options = {f"#{i['number']} - {i['title']}": i for i in issues}
        selected_issue_title = st.selectbox("Select an issue to work on:", list(issue_options.keys()))
        selected_issue = issue_options[selected_issue_title]

        if st.button("📝 Generate My First PR Plan", key="btn_pr_plan"):
            with st.spinner("Generating step-by-step Git instructions and PR draft..."):
                pr_prompt = f"""Create a beginner-friendly Pull Request step-by-step execution plan.
Repo: {info.get('full_name')}
Target Issue: #{selected_issue['number']} - {selected_issue['title']}
URL: {selected_issue['html_url']}
Main Language: {info.get('language')}

Provide:
1. Exact Git commands to fork, clone, create a feature branch (`fix-issue-{selected_issue['number']}`), and install dependencies.
2. Suggested files to explore and inspect.
3. Git commit message following Conventional Commits format.
4. Git push command.
5. Ready-to-copy PR title and markdown description template linking to the issue."""

                if st.session_state.get("demo_mode", False) or not ui_do_key:
                    pr_plan = f"""### 🚀 Git Workflow Plan for #{selected_issue['number']}

#### 1. Terminal Commands:
```bash
# Fork the repo on GitHub first, then clone your fork:
git clone https://github.com/<your-username>/{repo}.git
cd {repo}

# Create and switch to a dedicated feature branch
git checkout -b fix-issue-{selected_issue['number']}

# Install dependencies (see project README)
```

#### 2. Make Your Fix & Test
Inspect relevant files and run the test suite to verify your changes.

#### 3. Commit & Push
```bash
git add .
git commit -m "fix: resolve #{selected_issue['number']} - {selected_issue['title']}"
git push origin fix-issue-{selected_issue['number']}
```

#### 4. Pull Request Template:
- **Title:** `fix: resolve issue #{selected_issue['number']}`
- **Description:**
  > Closes #{selected_issue['number']}
  >
  > ### Changes Made
  > - Fixed {selected_issue['title']}
  > - Added/updated tests where applicable
"""
                else:
                    pr_plan = ask_llm(
                        prompt=pr_prompt,
                        model=active_model,
                        base_url=ui_base_url,
                        api_key=ui_do_key
                    )

                st.markdown(pr_plan)
