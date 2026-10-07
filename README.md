# ARC Interactive Agent

This project studies within-episode learning in a synthetic grid mock and an observation-driven ARC-AGI-3 agent.
The synthetic mock was built by the same team and is not independent evidence of competition performance.
On the registered real-game holdout, random completed a level in 2/120 episodes and click-exploration in 0/120.
The exact paired McNemar p-value was 0.5; neither policy won a game.
Click-exploration showed no advantage over random; low power makes this result inconclusive.

## Environment setup

Use Python 3.11 (ARC1) for the synthetic mock and its regression suite:

```powershell
$ARC1_PYTHON = "<ANACONDA_ROOT>\envs\ARC1\python.exe"
& $ARC1_PYTHON -m pip install -e ".[test]"
```

Use Python 3.12 (ARC312) for real-game evaluation:

```powershell
$ARC312_PYTHON = "<ANACONDA_ROOT>\envs\ARC312\python.exe"
& $ARC312_PYTHON -m pip install -e ".[test]"
& $ARC312_PYTHON -m pip install --no-index --find-links "<KAGGLE_ARC_AGI_WHEELHOUSE>" "arc-agi==0.9.8"
```

Competition game files are **not included**. Obtain them from the ARC-AGI-3 Kaggle competition and set the directory containing the environment files:

```powershell
$env:ARC_ENVIRONMENTS_DIR = "<PATH_TO_KAGGLE_ENVIRONMENT_FILES>"
```

Real-game evaluation also needs the authorized development split JSON and game-survey CSV. They are evaluation data and are not part of this code-only export; pass their locations with `--split-file` and `--survey-file`.

## Tests

Run the complete available suite on each interpreter:

```powershell
& $ARC1_PYTHON -m pytest -q
& $ARC312_PYTHON -m pytest -q
```

Tests that require captured competition fixtures or excluded writeup/evaluation data skip when those inputs are absent; fixture-independent tests still run.

## Reproducing the synthetic ablations

These frozen synthetic conditions do not use competition files:

```powershell
& $ARC1_PYTHON -m arc_agent.eval.ablation --dev-only --difficulty large-tight-budget --dev-start 950000 --episodes 200 --output-prefix <OUTPUT_PREFIX_LARGE>
& $ARC1_PYTHON -m arc_agent.eval.ablation --dev-only --difficulty key-door --dev-start 951000 --episodes 200 --output-prefix <OUTPUT_PREFIX_KEY_DOOR>
```

## Running a real-game development sweep

With ARC312, `ARC_ENVIRONMENTS_DIR`, and the separately obtained development registries set, run the matched budgets below. Each command evaluates random and click-exploration on five seeds per registered game. Choose new output prefixes for each run.

```powershell
& $ARC312_PYTHON -m arc_agent.eval.arc_runner --split development --split-file <DEVELOPMENT_SPLIT_JSON> --survey-file <GAME_SURVEY_CSV> --environments-dir $env:ARC_ENVIRONMENTS_DIR --agents random click-exploration --seed-start 20262000 --seeds-per-game 5 --max-actions 50 --output-prefix <OUTPUT_PREFIX_50>
& $ARC312_PYTHON -m arc_agent.eval.arc_runner --split development --split-file <DEVELOPMENT_SPLIT_JSON> --survey-file <GAME_SURVEY_CSV> --environments-dir $env:ARC_ENVIRONMENTS_DIR --agents random click-exploration --seed-start 20262000 --seeds-per-game 5 --max-actions 150 --output-prefix <OUTPUT_PREFIX_150>
& $ARC312_PYTHON -m arc_agent.eval.arc_runner --split development --split-file <DEVELOPMENT_SPLIT_JSON> --survey-file <GAME_SURVEY_CSV> --environments-dir $env:ARC_ENVIRONMENTS_DIR --agents random click-exploration --seed-start 20262000 --seeds-per-game 5 --max-actions 400 --output-prefix <OUTPUT_PREFIX_400>
```

## Analyzing held-out CSVs

This command consumes separately obtained episode CSVs; it does not rerun games:

```powershell
& $ARC312_PYTHON -m scripts.analyze_heldout --click-random-csv <RANDOM_CLICK_EPISODES_CSV> --frontier-csv <FRONTIER_EPISODES_CSV> --output-dir <ANALYSIS_OUTPUT_DIRECTORY>
```

## Result interpretation

The development sweep showed partial level progress but no wins. On the held-out set, random had a 1.7% episode-level completion rate and click-exploration had 0%; the two-sided exact McNemar p-value was 0.5. There were only two discordant pairs, so the evaluation has low power and does not establish equivalence.
