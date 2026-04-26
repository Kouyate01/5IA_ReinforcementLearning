$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
$env:PYTHONIOENCODING = "utf-8"
$PYTHON = "C:\Program Files\Python39\python.exe"

Write-Host "=== DEBUT ENTRAINEMENTS REINFORCE ===" -ForegroundColor Cyan
Write-Host "Total estime : ~3h30" -ForegroundColor Yellow

# ── LineWorld (100 000 episodes) ────────────────────────────────────────────
Write-Host "`n[1/12] REINFORCE - LineWorld 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce --env line_world --episodes 100000 --seeds 42 --config configs/config_reinforce.yaml

Write-Host "`n[2/12] REINFORCE Mean Baseline - LineWorld 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_mb --env line_world --episodes 100000 --seeds 42 --config configs/config_reinforce_mb.yaml

Write-Host "`n[3/12] REINFORCE Critic - LineWorld 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_critic --env line_world --episodes 100000 --seeds 42 --config configs/config_reinforce_critic.yaml

# ── GridWorld (10 000 episodes - env de test, on economise du temps) ────────
Write-Host "`n[4/12] REINFORCE - GridWorld 10K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce --env grid_world --episodes 10000 --seeds 42 --config configs/config_reinforce.yaml

Write-Host "`n[5/12] REINFORCE Mean Baseline - GridWorld 10K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_mb --env grid_world --episodes 10000 --seeds 42 --config configs/config_reinforce_mb.yaml

Write-Host "`n[6/12] REINFORCE Critic - GridWorld 10K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_critic --env grid_world --episodes 10000 --seeds 42 --config configs/config_reinforce_critic.yaml

# ── TicTacToe (100 000 episodes) ────────────────────────────────────────────
Write-Host "`n[7/12] REINFORCE - TicTacToe 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce --env tictactoe --episodes 100000 --seeds 42 --config configs/config_reinforce.yaml

Write-Host "`n[8/12] REINFORCE Mean Baseline - TicTacToe 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_mb --env tictactoe --episodes 100000 --seeds 42 --config configs/config_reinforce_mb.yaml

Write-Host "`n[9/12] REINFORCE Critic - TicTacToe 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_critic --env tictactoe --episodes 100000 --seeds 42 --config configs/config_reinforce_critic.yaml

# ── Bobail (100 000 episodes) ────────────────────────────────────────────────
Write-Host "`n[10/12] REINFORCE - Bobail 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce --env bobail --episodes 100000 --seeds 42 --config configs/config_reinforce.yaml

Write-Host "`n[11/12] REINFORCE Mean Baseline - Bobail 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_mb --env bobail --episodes 100000 --seeds 42 --config configs/config_reinforce_mb.yaml

Write-Host "`n[12/12] REINFORCE Critic - Bobail 100K..." -ForegroundColor Green
& $PYTHON train.py --agent reinforce_critic --env bobail --episodes 100000 --seeds 42 --config configs/config_reinforce_critic.yaml

Write-Host "`n=== TOUS LES ENTRAINEMENTS TERMINES ===" -ForegroundColor Cyan
Write-Host "Les modeles sont dans : saved_models/" -ForegroundColor Yellow
Write-Host "Les metriques sont dans : results/" -ForegroundColor Yellow
