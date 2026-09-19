# Putting this repository on GitHub, click by click

Written 2026-09-19 for a Windows 11 machine with Git installed. Two routes are given. Route A uses the GitHub website plus a terminal. Route B uses the `gh` command line tool and skips the website.

## Before either route: the email block

GitHub can reject a push with error `GH007` when a commit carries your real email address and your account has "Block command line pushes that expose my email" switched on. This machine has hit that before. Do this once:

1. Open https://github.com/settings/emails in a browser.
2. Under "Keep my email addresses private", copy the address that looks like `12345678+yourname@users.noreply.github.com`.
3. In a terminal inside the repository folder, run:

```bash
git config user.email "12345678+yourname@users.noreply.github.com"
```

```bash
git config user.name "Your Name"
```

If you already made a commit with the wrong address, fix it before pushing:

```bash
git commit --amend --reset-author --no-edit
```

## Route A: website plus terminal

### Step 1. Turn the folder into a repository

Open a terminal (PowerShell or Git Bash) in `C:\Users\cld-main\Desktop\github-projects\jev-test`, then run each line:

```bash
git init -b main
```

```bash
git add .
```

```bash
git commit -m "Pre-register judgment spec v1 and repository scaffold"
```

### Step 2. Create the empty repository on GitHub

1. Go to https://github.com and sign in.
2. Click the plus sign in the top right corner, then click "New repository".
3. Owner: leave as your account.
4. Repository name: type `jev-test`.
5. Description: paste this: `Can a small local model do trustworthy web search-and-answer when a decision model (Jev) makes every call? Pre-registered benchmark with Gemma 4 E2B, SearXNG, and MemPalace.`
6. Choose "Public".
7. Leave "Add a README file" unchecked. Leave ".gitignore" as "None". Leave "License" as "None". The folder already has all three.
8. Click "Create repository".
9. On the page that appears, find the block titled "...or push an existing repository from the command line" and copy the remote URL shown. It looks like `https://github.com/yourname/jev-test.git`.

### Step 3. Connect and push

Back in the terminal:

```bash
git remote add origin https://github.com/yourname/jev-test.git
```

```bash
git push -u origin main
```

A browser window may open asking you to sign in to GitHub. Approve it. The push finishes in the terminal.

### Step 4. Fill in the About box and topics

1. Open the repository page on GitHub.
2. On the right side, next to "About", click the gear icon.
3. Website: leave empty for now. You will put the Hugging Face Space URL here later.
4. Topics: type each of these and press Enter after each: `jev`, `typesafe`, `system-one`, `rag`, `small-language-models`, `gemma`, `searxng`, `mempalace`, `benchmark`, `hallucination`.
5. Tick "Releases" off and "Packages" off if you prefer a cleaner sidebar. Leave "Deployments" off.
6. Click "Save changes".

### Step 5. Settings worth setting now

1. Click "Settings" on the repository page (the tab on the far right of the top bar).
2. Under "General", scroll to "Features". Tick "Issues". Tick "Discussions" if you want a place for people to argue about results; that is where Reddit traffic tends to land. Leave "Wikis" and "Projects" unticked.
3. Under "General", scroll to "Pull Requests". Tick "Automatically delete head branches".
4. In the left menu, click "Branches". Click "Add classic branch protection rule". Branch name pattern: `main`. Tick "Require a pull request before merging". Click "Create". This stops a stray push from overwriting the pre-registered spec.
5. In the left menu, click "Secrets and variables", then "Actions". Do not add any secrets yet. The Jev key never goes here. When the Space needs it, it goes in Hugging Face's Space secrets as a bring-your-own-key field, not in GitHub.

### Step 6. Tag the pre-registration

The whole credibility argument rests on the spec being committed before any run. Make that visible:

```bash
git tag -a v1-prereg -m "Judgment spec v1 pre-registered before any benchmark run"
```

```bash
git push origin v1-prereg
```

On GitHub, click "Releases" on the right side, click "Draft a new release", choose tag `v1-prereg`, title it `v1 pre-registration`, paste the four bars from the spec's section 11 into the description, and click "Publish release". The release page carries a timestamp nobody can edit.

## Route B: the gh command line tool

Only if `gh` is installed and signed in (`gh auth status` shows your account).

```bash
git init -b main
```

```bash
git add .
```

```bash
git commit -m "Pre-register judgment spec v1 and repository scaffold"
```

```bash
gh repo create jev-test --public --source=. --remote=origin --push --description "Can a small local model do trustworthy web search-and-answer when a decision model (Jev) makes every call? Pre-registered benchmark with Gemma 4 E2B, SearXNG, and MemPalace."
```

```bash
gh repo edit --add-topic jev --add-topic typesafe --add-topic system-one --add-topic rag --add-topic small-language-models --add-topic gemma --add-topic searxng --add-topic mempalace --add-topic benchmark --add-topic hallucination
```

```bash
git tag -a v1-prereg -m "Judgment spec v1 pre-registered before any benchmark run" && git push origin v1-prereg
```

```bash
gh release create v1-prereg --title "v1 pre-registration" --notes "Coverage >= 0.50; hallucination when attempting <= 0.10; truthfulness gain over arm B >= +0.15; citation support >= 0.90. Memory track: judge requests <= 0.5x pass one; hallucination <= pass one + 0.02."
```

Branch protection and the Issues/Discussions toggles still need the website; see Route A steps 4 and 5.

## What not to commit, ever

`.gitignore` already covers these, but know why:

- `.env` holds the Jev key and the grading key.
- `json_cache/` can hold thousands of judge responses. Ship it as a release asset if you want reproducibility, not as tracked files.
- The MemPalace palace directory. It is the frozen snapshot. Tar it and attach it to a release.
- Dataset files. Loaders download them; licenses differ.

## Later: the Hugging Face Space

When there are results to show, create the Space from the Hugging Face side (New, then Space, SDK "Gradio", hardware "ZeroGPU" or "CPU basic"), then in this repository add a GitHub Action that pushes `space/` and `results/` to the Space on every release. That step is not part of the first push.
