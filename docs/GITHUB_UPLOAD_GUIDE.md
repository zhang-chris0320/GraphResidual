# GitHub upload guide

This workspace is initialized as a local Git repository only. No remote repository is created automatically.

After creating an empty GitHub repository and reviewing the files, run:

    cd <release-directory>
    git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
    git branch -M main
    git push -u origin main

Before pushing, confirm that raw data, datasets, logs, credentials, tokens, SSH material, and checkpoint binaries are not staged. The included .gitignore excludes those classes by default.
