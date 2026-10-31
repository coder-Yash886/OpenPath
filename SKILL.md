---
name: repo-guide
description: Explains any public GitHub repository for open-source beginners in simple Hinglish, Hindi, or English. Extracts README, file structure, and good-first-issues, then uses an open-weight model on DigitalOcean Serverless Inference to generate an onboarding map and PR contribution plan.
---

# RepoGuide Agent Skill

This skill explains public GitHub repositories for beginners looking to make their first open-source contributions.

## When to Use

Use this skill when:
- A user shares a GitHub repository link and wants to know what it does or how to contribute.
- A beginner developer is overwhelmed by a codebase and needs a structured walkthrough.
- The user asks for beginner-friendly tasks or "good first issue" tickets in a repository.
- Generating a step-by-step Git contribution plan (fork, clone, branch, commit, push, PR).

## Workflow

1. **Parse Repository URL**:
   Extract `{owner}` and `{repo}` from the GitHub URL (e.g., `https://github.com/fastapi/fastapi` -> owner: `fastapi`, repo: `fastapi`).

2. **Fetch Repository Context via GitHub REST API**:
   - Repository Metadata: `GET https://api.github.com/repos/{owner}/{repo}`
   - README content: `GET https://api.github.com/repos/{owner}/{repo}/readme` (Base64 decode)
   - Git Tree: `GET https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1`
   - Good First Issues: `GET https://api.github.com/repos/{owner}/{repo}/issues?labels=good+first+issue&state=open&per_page=5` (exclude PRs)

3. **Inference with Open-Weight LLM via DigitalOcean Gradient AI**:
   - Endpoint: `https://inference.do-ai.run/v1`
   - Model: `alibaba-qwen3-32b` (or `meta-llama/llama-3.1-70b-instruct`)
   - Send truncated README (max 6000 chars) and file paths (max 250 paths).

4. **Output Format**:
   Deliver a markdown response with these 4 core sections:
   - `## What this project does` (3-5 simple lines)
   - `## Where things live` (concise directory and file architecture map)
   - `## How to run it locally` (tested setup and launch instructions)
   - `## Where to start contributing` (up to 3 beginner tasks based on open tickets)

5. **Generate PR Contribution Plan**:
   For any selected issue, outline concrete Git commands (`git checkout -b`, branch conventions, testing, and PR description template).
