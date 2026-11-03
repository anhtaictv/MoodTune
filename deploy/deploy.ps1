<#
  Deploy MoodTune từ thư mục checkout của runner vào C:\moodtune (nơi PM2 + IIS đang phục vụ).

  - Chỉ copy file có trong git và khác nội dung với bản đang chạy; file cũ được backup
    vào backend\backups\deploy-<timestamp>\ trước khi ghi đè.
  - Trạng thái model (weights*, dynamic_*) do online learning ghi liên tục lúc chạy ->
    mặc định KHÔNG ghi đè, trừ khi truyền -DeployModel (vd sau khi retrain offline).
  - Restart PM2 rồi chờ /api/health; nếu fail -> khôi phục backup và restart lại.

  Ví dụ chạy tay (xem trước, không ghi gì):  .\deploy\deploy.ps1 -Source . -DryRun
#>
param(
  [Parameter(Mandatory = $true)][string]$Source,
  [string]$Target = "C:\moodtune",
  [switch]$DeployModel,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PM2_APP = "moodtune-backend"
$HEALTH_URL = "http://127.0.0.1:5005/api/health"
$HEALTH_TIMEOUT_SEC = 60
$DEPLOY_DIRS = @("backend", "frontend")
# File runtime state: model tự ghi trong lúc chạy, bản trong git thường cũ hơn bản live.
$MODEL_STATE_FILES = @(
  "backend\weights.npz",
  "backend\weights_meta.json",
  "backend\weights_replay.json",
  "backend\dynamic_lexicon.json",
  "backend\dynamic_vocab.json"
)

$Source = (Resolve-Path $Source).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $Target "backend\backups\deploy-$stamp"
$python = if ($env:MOODTUNE_PYTHON) { $env:MOODTUNE_PYTHON } else { "C:\Users\Administrator\AppData\Local\Programs\Python\Python314\python.exe" }

function Get-ChangedFiles {
  $changed = @()
  # Resolve-Path -Relative thay vì cắt chuỗi: $Source có thể là đường dẫn 8.3 (ADMINI~1)
  # trong khi FullName của Get-ChildItem là đường dẫn dài.
  Push-Location $Source
  try {
    foreach ($dir in $DEPLOY_DIRS) {
      Get-ChildItem $dir -Recurse -File | Where-Object { $_.FullName -notmatch '\\__pycache__\\' } | ForEach-Object {
        $rel = (Resolve-Path -Relative $_.FullName) -replace '^\.\\', ''
        $isState = $MODEL_STATE_FILES -contains $rel
        $dest = Join-Path $Target $rel
        if ($isState -and -not $DeployModel -and (Test-Path $dest)) { return }
        if ((Test-Path $dest) -and ((Get-FileHash $dest).Hash -eq (Get-FileHash $_.FullName).Hash)) { return }
        $changed += $rel
      }
    }
  } finally { Pop-Location }
  return $changed
}

function Copy-WithBackup([string[]]$files) {
  foreach ($rel in $files) {
    $dest = Join-Path $Target $rel
    if (Test-Path $dest) {
      $bak = Join-Path $backupDir $rel
      New-Item -ItemType Directory -Force -Path (Split-Path $bak) | Out-Null
      Copy-Item $dest $bak -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Copy-Item (Join-Path $Source $rel) $dest -Force
  }
}

function Restore-Backup([string[]]$files) {
  foreach ($rel in $files) {
    $bak = Join-Path $backupDir $rel
    $dest = Join-Path $Target $rel
    if (Test-Path $bak) { Copy-Item $bak $dest -Force }
    elseif (Test-Path $dest) { Remove-Item $dest -Force }  # file mới thêm trong lần deploy này
  }
}

function Wait-Healthy {
  $deadline = (Get-Date).AddSeconds($HEALTH_TIMEOUT_SEC)
  while ((Get-Date) -lt $deadline) {
    try {
      $res = Invoke-WebRequest -Uri $HEALTH_URL -UseBasicParsing -TimeoutSec 5
      if ($res.StatusCode -eq 200) { return $true }
    } catch { }
    Start-Sleep -Seconds 2
  }
  return $false
}

function Invoke-Pm2([string[]]$pm2Args) {
  pm2 @pm2Args
  if ($LASTEXITCODE -ne 0) { throw "pm2 $($pm2Args -join ' ') failed (exit $LASTEXITCODE)" }
}

$changed = @(Get-ChangedFiles)
if ($changed.Count -eq 0) {
  Write-Host "Không có file nào thay đổi - bỏ qua deploy."
  exit 0
}
Write-Host "File sẽ deploy ($($changed.Count)):"
$changed | ForEach-Object { Write-Host "  $_" }
if ($DryRun) { exit 0 }

$backendChanged = @($changed | Where-Object { $_ -like "backend\*" }).Count -gt 0
$reqChanged = $changed -contains "backend\requirements.txt"

# Frontend là file tĩnh IIS phục vụ trực tiếp -> không cần restart.
if (-not $backendChanged) {
  Copy-WithBackup $changed
  Write-Host "Chỉ frontend thay đổi - đã cập nhật, không restart backend. Backup: $backupDir"
  exit 0
}

# Python giữ khoá file .pyd khi đang chạy -> phải stop trước khi pip cài lại.
if ($reqChanged) { Invoke-Pm2 @("stop", $PM2_APP) }
Copy-WithBackup $changed
if ($reqChanged) {
  & $python -m pip install --disable-pip-version-check -r (Join-Path $Target "backend\requirements.txt")
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip install lỗi - khôi phục bản cũ"
    Restore-Backup $changed
    Invoke-Pm2 @("restart", $PM2_APP)
    throw "pip install failed"
  }
}

Invoke-Pm2 @("restart", $PM2_APP)
if (Wait-Healthy) {
  Invoke-Pm2 @("save")
  Write-Host "Deploy OK - $HEALTH_URL trả về 200. Backup: $backupDir"
  exit 0
}

Write-Warning "Health check fail sau $HEALTH_TIMEOUT_SEC s - rollback"
Restore-Backup $changed
Invoke-Pm2 @("restart", $PM2_APP)
if (-not (Wait-Healthy)) { Write-Warning "Rollback xong nhưng backend vẫn chưa healthy - cần kiểm tra tay (pm2 logs $PM2_APP)" }
throw "Deploy failed health check, đã rollback"
