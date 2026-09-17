param(
    [Parameter(Mandatory = $true)]
    [string]$MappingDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [switch]$Plan10,
    [string]$Manifest
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

python (Join-Path $scriptDir "build_ptz_pixel_motion_relation.py") `
    --mapping-dir $MappingDir `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Plan10) {
    if (-not $Manifest) {
        $Manifest = Join-Path $OutputDir "center_only_batch_manifest.json"
    }
    python (Join-Path $scriptDir "plan_ptz_center_only_batch.py") `
        --relation-dir $OutputDir `
        --mapping-dir $MappingDir `
        --output $Manifest
    exit $LASTEXITCODE
}
