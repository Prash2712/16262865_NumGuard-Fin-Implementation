# GitHub Push Instructions

After extracting this package on macOS or Linux:

```bash
cd NumGuard-Fin-GitHub-Complete
git init
git branch -M main
git config user.name "Prasanth Balisetty"
git config user.email "YOUR_GITHUB_EMAIL"
git add .
git status
git commit -m "Add complete NumGuard-Fin dissertation implementation and evidence"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before `git commit`, review `git status` and confirm that raw FinQA data, model caches, secrets and temporary runtime files are absent.

The complete package intentionally tracks the selected historical and evidence ZIP archives under `development_history/` and `evidence_archives/`. Generated rerun packages remain ignored until manually reviewed.

For university marking, consider keeping the repository private because the dissertation contains the student's name, student identifier and ethics reference.
