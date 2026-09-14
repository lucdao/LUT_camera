param(
    [switch]$Capture,
    [switch]$Execute,
    [string]$RunRoot,
    [string]$WideUrl = $env:WIDE_RTSP_URL,
    [string]$PtzHost = "192.168.1.8",
    [int]$PtzPort = 80,
    [string]$PtzUser = "admin",
    [int]$WideCount = 8,
    [int]$PanCount = 30,
    [int]$TiltCount = 6
)

$arguments = @(".\run_two_stage_pipeline.py")
if ($Capture) { $arguments += "--capture" }
if ($Execute) { $arguments += "--execute" }
if ($RunRoot) { $arguments += @("--run-root", $RunRoot) }
if ($WideUrl) { $env:WIDE_RTSP_URL = $WideUrl }
$arguments += @(
    "--ptz-host", $PtzHost,
    "--ptz-port", $PtzPort,
    "--ptz-user", $PtzUser,
    "--wide-count", $WideCount,
    "--pan-count", $PanCount,
    "--tilt-count", $TiltCount,
    "--continue-on-error"
)

python @arguments
exit $LASTEXITCODE
