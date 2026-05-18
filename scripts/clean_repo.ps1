# Clean repo helper script (PowerShell)
# Usage: ./scripts/clean_repo.ps1 -Confirm:$false
param(
    [switch]$Confirm = $true
)

Write-Output "Cleaning repository..."
# Remove Jupyter checkpoints
Get-ChildItem -Path . -Recurse -Directory -Filter '.ipynb_checkpoints' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Remove __pycache__
Get-ChildItem -Path . -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item $_.FullName -Recurse -Force }
# Remove temporary results
if (Test-Path .\results\tmp) { Remove-Item .\results\tmp -Recurse -Force }
Write-Output "Done."
