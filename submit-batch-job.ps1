param(
  [string]$User = "ayvazoglu",
  [string]$RemoteHost = "remote.cip.ifi.lmu.de",
  [string]$RemoteDir = "~/facial-emotion-classifier",
  [string]$SlurmScript = "slurm/train.slurm"
)

# 1) Sync code to cluster (excluding big/local stuff)
rsync -av --delete `
  --exclude ".git" `
  --exclude ".venv" `
  --exclude "data" `
  --exclude "logs" `
  --exclude ".vscode" `
  --exclude "checkpoints" `
  . "$User@${RemoteHost}:$RemoteDir/"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }


ssh "$User@$RemoteHost" "cd $RemoteDir && mkdir -p logs checkpoints && sbatch $SlurmScript"