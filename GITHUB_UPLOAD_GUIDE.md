# Upload to GitHub

From inside the repository folder:

```bash
git init
git add .
git commit -m "Initial ChestX-ray14 classification pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

Before pushing, verify that `git status` does **not** show dataset files, `.env`, API keys, or model checkpoints.

If a secret was ever committed, deleting it in a later commit is not sufficient; rotate/revoke the credential and, if necessary, remove it from Git history.
