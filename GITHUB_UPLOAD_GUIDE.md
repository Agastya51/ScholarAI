# How to put ScholarAI on GitHub — start to finish

Written assuming you've never used GitHub before. Follow it top to bottom; it takes about 30 minutes.

> **This file is just instructions for you — delete it before you push, or keep it, it doesn't matter. It won't hurt anything.**

---

## First: what is GitHub, and what's a "username"?

GitHub is a website where programmers store their code publicly. Think of it as Google Drive for code, except everyone can see it and recruiters actually go look at it.

When you sign up you pick a **username** — a short handle like `agastyakundu` or `akundu23`. Your profile then lives at `github.com/your-username`, and every project you upload lives at `github.com/your-username/project-name`. That URL is what you put on your resume.

A **repository** (everyone says "repo") is one project folder. You'll create one called `ScholarAI`.

---

## Step 1 — Create your GitHub account

1. Go to **[github.com/signup](https://github.com/signup)**.
2. Use an email you'll keep after graduation. Your college email works, but add a personal one as a backup later.
3. **Pick your username carefully — it goes on your resume.** Good: `agastyakundu`, `agastya-kundu`, `akundu-dev`. Avoid: `coolcoder2004`, `agastya12345`.
4. Verify the email GitHub sends you.

**While you're there, spend 5 minutes on your profile:** add a real photo, your name, "ECE @ LNMIIT" as the bio, and your LinkedIn. Recruiters click through.

---

## Step 2 — Install Git on your computer

Git is the tool that uploads code to GitHub. GitHub is the website; Git is the program.

- **Windows:** download from [git-scm.com/download/win](https://git-scm.com/download/win), run the installer, click Next on everything. Defaults are fine.
- **Mac:** open Terminal and type `git --version`. If it's not installed, macOS offers to install it — say yes.
- **Linux:** `sudo apt install git`

Check it worked. Open **Command Prompt** (Windows — press Start, type `cmd`) or **Terminal** (Mac/Linux) and run:

```bash
git --version
```

You should see something like `git version 2.43.0`. If you get "command not found", the install didn't work — restart your computer and try again.

---

## Step 3 — Tell Git who you are

One-time setup. Run these two commands, using the email you signed up with:

```bash
git config --global user.name "Agastya Kundu"
git config --global user.email "your-email@example.com"
```

---

## Step 4 — 🚨 Check your API key is not in the code

**Do this before anything else. This is the single most common way students get burned.**

If your Groq API key is written anywhere inside a `.py` file, and you push it to a public repo, bots will find it within minutes and drain your account. Once a key is in Git history, deleting it later does *not* remove it.

Open every `.py` file and search for `gsk_`. Your key looks like `gsk_abc123XYZ...`.

- If you find it: **delete the key from the code**, and make sure `config.py` reads it from the environment instead. In your version it already does this correctly:
  ```python
  GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
  ```
  That's the right pattern — the key lives in a `.env` file, and `.env` is listed in `.gitignore` so Git ignores it.

- Also check: do you have a `.env` file in your folder? **Good — keep it, but never commit it.** The `.gitignore` I've included takes care of that automatically.

If you *have* already leaked a key at some point: go to [console.groq.com/keys](https://console.groq.com/keys), delete the old key, and generate a new one. Takes 10 seconds.

---

## Step 5 — Put the new files into your project folder

Download the ZIP I sent you and copy these into your existing project folder, replacing the old ones:

| File | What it does |
| --- | --- |
| `README.md` | Replaces your old README — this is the page recruiters read |
| `.gitignore` | Tells Git to never upload your `.env`, venv, or caches |
| `.env.example` | Shows others what key they need, without revealing yours |
| `LICENSE` | MIT license |
| `requirements.txt` | Cleaned up — removed `flask` and `reportlab` (unused) |
| `docs/ARCHITECTURE.md` | Deep technical doc |
| `docs/BTP_Report.pdf` | Your project report, linked from the README |

Your folder should end up looking like this:

```
ScholarAI/
├── app.py
├── config.py
├── utils.py
├── graphs.py
├── api_search.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── .env              ← your real key, NOT uploaded
└── docs/
    ├── ARCHITECTURE.md
    └── BTP_Report.pdf
```

> **Windows tip:** files starting with a dot (`.gitignore`, `.env.example`) may look invisible. In File Explorer, go to View → Show → Hidden items.

---

## Step 6 — Create the repository on GitHub

1. Go to **[github.com/new](https://github.com/new)**.
2. **Repository name:** `ScholarAI`
3. **Description:** `Multimodal RAG + Agentic AI research paper analyser — summarisation, vision figure analysis, grounded Q&A, and 4-source literature comparison. Built with LangGraph, FAISS and Groq.`
4. Select **Public**. (Private repos can't be seen by recruiters — the whole point is visibility.)
5. **Do NOT tick** "Add a README", "Add .gitignore" or "Choose a license" — you already have all three. Ticking them causes a merge conflict on your first push, which is annoying to fix.
6. Click **Create repository**.

You'll land on a page showing some commands. Ignore it for now — the next step covers it.

---

## Step 7 — Upload your code

Open Command Prompt / Terminal and navigate to your project folder:

```bash
# Windows example
cd C:\Users\YourName\Desktop\ScholarAI

# Mac/Linux example
cd ~/Desktop/ScholarAI
```

> Shortcut: type `cd ` (with the space), then drag the folder from your file explorer into the terminal window — it fills in the path for you.

Now run these commands **one at a time**, in order:

```bash
git init
```
Turns the folder into a Git repository.

```bash
git add .
```
Stages every file (except the ones `.gitignore` excludes).

```bash
git status
```
**Stop and read this output.** It lists everything about to be uploaded. **`.env` must NOT appear in this list.** If it does, `.gitignore` isn't in the folder — go back to Step 5. Do not continue until `.env` is absent.

```bash
git commit -m "ScholarAI: multimodal RAG research paper analyser"
```
Saves a snapshot locally.

```bash
git branch -M main
```
Names the main branch `main` (GitHub's default).

```bash
git remote add origin https://github.com/YOUR-USERNAME/ScholarAI.git
```
**Replace `YOUR-USERNAME` with your actual GitHub username.** Points your local folder at the online repo.

```bash
git push -u origin main
```
Uploads everything.

**On the push, a browser window opens asking you to sign in to GitHub.** Sign in and authorise. If you instead get a password prompt in the terminal, your GitHub *password* will not work — you need a Personal Access Token:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) → Generate new token (classic)
2. Note: `laptop`, Expiration: 90 days, tick the **`repo`** checkbox
3. Generate, **copy the token immediately** (you can't see it again)
4. Paste it when the terminal asks for a password (it won't show characters as you type — that's normal)

---

## Step 8 — Check your work

Go to `github.com/YOUR-USERNAME/ScholarAI`. You should see:

- ✅ All your `.py` files
- ✅ The README rendering below the file list, with badges at the top
- ✅ The architecture diagrams rendering as actual flowcharts (GitHub renders Mermaid natively)
- ❌ **No `.env` file** — confirm this by eye

Click into `docs/ARCHITECTURE.md` and confirm those diagrams render too.

---

## Step 9 — Add screenshots (do not skip this)

A README with screenshots gets read. One without gets scrolled past. This is 10 minutes of work for a disproportionate payoff.

1. Run your app: `streamlit run app.py`
2. Upload a paper and let each tab finish.
3. Screenshot the **Summary**, **Figures**, and **Compare** tabs. (Windows: `Win + Shift + S`. Mac: `Cmd + Shift + 4`.)
4. Make a folder `docs/images/` and save them as `summary.png`, `figures.png`, `compare.png`.
5. Open `README.md`, find the Screenshots section, and replace the placeholder table with:

```markdown
### Summary tab
![Summary](docs/images/summary.png)

### Figure analysis
![Figures](docs/images/figures.png)

### Comparative study
![Compare](docs/images/compare.png)
```

6. Push the changes:

```bash
git add .
git commit -m "Add screenshots"
git push
```

That last three-command sequence is how you push *any* future change. It's the only Git you'll need day to day.

---

## Step 10 — Polish the repo page

On your repo page, click the ⚙️ gear icon next to "About" on the right side:

- **Description:** paste the one from Step 6
- **Topics:** add `rag`, `llm`, `langgraph`, `agentic-ai`, `streamlit`, `faiss`, `multimodal`, `groq`, `research-tools`, `python` — these make your repo findable and signal what you work with at a glance
- **Website:** leave blank for now, or add your live Streamlit URL if you deploy (see below)

Then **pin the repo to your profile:** go to `github.com/YOUR-USERNAME`, click "Customize your pins", and select ScholarAI. Pinned repos are the first thing anyone sees.

---

## Optional but high-value: deploy it live

A working link beats a code listing every time. Streamlit Community Cloud hosts it free:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in **with GitHub**.
2. Click "New app", pick your `ScholarAI` repo, `main` branch, `app.py` as the main file.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
   (This is Streamlit's secret store — it does not go into your repo.)
4. Deploy. First build takes ~5 minutes while it installs torch.
5. Add the resulting URL to the "Website" field in your repo's About section, and as a "**🔗 Live demo**" link at the top of your README.

> Note: the free tier has ~1 GB of RAM. `torch` + `sentence-transformers` is close to that ceiling. If the build fails on memory, that's the cause — it's a known constraint, not a bug in your code.

---

## Putting it on your resume

Under a **Projects** section:

> **ScholarAI — Multimodal RAG Research Assistant** · [github.com/your-username/ScholarAI](https://github.com/your-username/ScholarAI)
> Built an agentic research-paper analyser using LangGraph state machines over four independent workflows. Implemented a RAG pipeline with FAISS and 384-D sentence-transformer embeddings for hallucination-free Q&A grounded in the source PDF. Integrated a vision-language model for automated figure analysis, and a multi-source retrieval layer querying arXiv, Semantic Scholar, OpenAlex and CrossRef with deduplication and abstract-quality filtering to generate five-dimension comparative literature analysis. Added two-tier LLM fallback for graceful degradation under rate limits.
> *Python, LangGraph, FAISS, Streamlit, Groq, PyMuPDF, sentence-transformers*

Two things make this land in an interview, so have answers ready:

**"Why RAG instead of just prompting the model with the paper?"** → Grounding. The retrieval step constrains the model to text that actually exists in the document, and it sidesteps the context-window limit for long papers.

**"Why LangGraph instead of a plain script?"** → Shared state across steps, fault isolation per node, and the ability to add or reorder steps without rewriting the pipeline. Point at the Compare graph — five sequential comparison nodes each getting their own full token budget, where one rate-limit failure costs one section instead of the whole analysis.

---

## When something goes wrong

| Error | Fix |
| --- | --- |
| `fatal: not a git repository` | You're in the wrong folder. `cd` into your project folder first. |
| `remote origin already exists` | `git remote remove origin`, then re-run the `git remote add` command. |
| `Updates were rejected... fetch first` | You ticked "Add a README" in Step 6. Run `git pull origin main --allow-unrelated-histories`, then push again. |
| `Support for password authentication was removed` | Use a Personal Access Token — see Step 8. |
| `.env` showed up in `git status` | `.gitignore` isn't in the folder, or is misnamed (`.gitignore.txt` is a common Windows mistake). Fix it, then `git rm --cached .env`. |
| Mermaid diagrams show as raw text | You're viewing the file in a plain text editor. They render on github.com. |
| Pushed the key by accident | Revoke it at [console.groq.com/keys](https://console.groq.com/keys) immediately and generate a new one. Then delete the repo on GitHub, remove the key from your files, and redo Step 7 from `git init`. |

---

## The three commands you'll use forever

```bash
git add .
git commit -m "what you changed"
git push
```

That's the whole day-to-day workflow. Everything else in this guide was one-time setup.
