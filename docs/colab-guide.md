# What you need to do in Google Colab

1. Create a Google Drive folder called `athlete-risk`.
2. Upload `notebooks/01_soccer_mon_colab.ipynb` from this repository to Colab.
3. Run every cell once with `DOWNLOAD_OBJECTIVE = False`. This is enough to inspect and use SoccerMon's small subjective data.
4. If a raw GPS field is genuinely needed, change the selected objective archive in the notebook and download **one** archive. Do not extract every archive. Colab runtimes are temporary; Drive must have enough space for both the ZIP and extracted subset.
5. Download or keep the generated `processed` CSV files in Drive. They are the only files this application needs for model training.
6. Put this repository in GitHub, then replace `<YOUR_GITHUB_REPOSITORY_URL>` in the README's Colab command with your repository URL.

You do not need to download the 99 GB dataset to your laptop.
