# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/aabd5c27-07f1-494f-b001-854c40db2b9f

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/aabd5c27-07f1-494f-b001-854c40db2b9f) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## Installation

Install the JavaScript dependencies with npm and the Python packages required for
testing using `pip`:

```bash
npm i
pip install -r requirements.txt
```

The optional `scripts/setup.sh` script and `Makefile` also install these
dependencies to speed up local setup and CI.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/aabd5c27-07f1-494f-b001-854c40db2b9f) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/tips-tricks/custom-domain#step-by-step-guide)

For an overview of the full data science pipeline, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Dashboard

After running the Prefect flows and generating models, you can launch the
Streamlit dashboard with:

```bash
streamlit run python/dashboard/app.py
```

The dashboard provides live signals, a model leaderboard, equity curves and
explainability plots.

## CLI usage

Run the Prefect flows directly from the command line:

```bash
# Run everything and launch the dashboard
python -m python.cli run-all --freq all --cleanup yes

# ``run-all`` checks available disk space before training and backtesting.
# If less than 5 GB remain it automatically invokes the cleanup flow.

# Or run individual steps
python -m python.cli ingest --freq day
python -m python.cli feature-build --freq day
python -m python.cli train-and-evaluate
python -m python.cli backtest
```


## Testing

Run the automated checks locally with:

```bash
npm run lint
pytest -q
```

Continuous integration runs the same commands via GitHub Actions (`.github/workflows/ci.yml`).
