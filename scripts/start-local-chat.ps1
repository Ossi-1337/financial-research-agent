[CmdletBinding()]
param(
    [string] $Model = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL",
    [string] $DockerImage = "ghcr.io/ggml-org/llama.cpp:server-cuda",
    [string] $ContainerName = "fra-llama-cpp",
    [int] $LlamaPort = 8080,
    [string] $AppHost = "127.0.0.1",
    [int] $AppPort = 8000,
    [int] $ContextSize = 32768,
    [int] $GpuLayers = 99,
    [int] $DockerRunWaitSeconds = 120,
    [int] $LlamaWaitSeconds = 1800,
    [int] $AppWaitSeconds = 60,
    [string] $HuggingFaceCache = (Join-Path $env:USERPROFILE ".cache\huggingface"),
    [switch] $RestartLlama,
    [switch] $RestartApp,
    [switch] $SkipDependencyInstall,
    [switch] $NoOpenBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot ".codex\logs"
$AppOutLog = Join-Path $LogDir "chat-ui.out.log"
$AppErrLog = Join-Path $LogDir "chat-ui.err.log"
$AppPidFile = Join-Path $LogDir "chat-ui.pid"
$DockerRunOutLog = Join-Path $LogDir "llama-docker-run.out.log"
$DockerRunErrLog = Join-Path $LogDir "llama-docker-run.err.log"
$script:ActiveContainerName = $ContainerName

function Test-CommandAvailable {
    param([string] $Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-ContainerRunning {
    param([string] $Name)

    $status = & docker ps -a --filter "name=^/$Name$" --format "{{.Status}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $statusText = (($status | Select-Object -First 1) -as [string]).Trim()
    if ($statusText -eq "") {
        return $null
    }

    return $statusText.StartsWith("Up", [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-ContainerWithTimeout {
    param(
        [string] $Name,
        [int] $TimeoutSeconds = 30
    )

    $process = Start-Process `
        -FilePath "docker" `
        -ArgumentList @("rm", "-f", $Name) `
        -WindowStyle Hidden `
        -PassThru

    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        return $false
    }

    return $process.ExitCode -eq 0
}

function New-FallbackContainerName {
    param([string] $BaseName)

    return "$BaseName-$([DateTime]::UtcNow.ToString("yyyyMMddHHmmss"))"
}

function Start-DockerRunWithTimeout {
    param([string[]] $Arguments)

    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Remove-Item -LiteralPath $DockerRunOutLog, $DockerRunErrLog -ErrorAction SilentlyContinue

    $process = Start-Process `
        -FilePath "docker" `
        -ArgumentList $Arguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $DockerRunOutLog `
        -RedirectStandardError $DockerRunErrLog `
        -PassThru

    if (-not $process.WaitForExit($DockerRunWaitSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $errorTail = ""
        if (Test-Path -LiteralPath $DockerRunErrLog) {
            $errorTail = (Get-Content -LiteralPath $DockerRunErrLog -Tail 40 | Out-String).Trim()
        }

        $message = (
            "Docker did not finish starting llama.cpp within $DockerRunWaitSeconds seconds. " +
            "Docker Desktop or the NVIDIA container runtime may be stuck. " +
            "Restart Docker Desktop and try again. Docker error log: $DockerRunErrLog"
        )
        if ($errorTail -ne "") {
            $message += "`nDocker error log tail:`n$errorTail"
        }
        throw $message
    }

    if ($process.ExitCode -ne 0) {
        $errorTail = ""
        if (Test-Path -LiteralPath $DockerRunErrLog) {
            $errorTail = (Get-Content -LiteralPath $DockerRunErrLog -Tail 40 | Out-String).Trim()
        }

        if ($errorTail -ne "") {
            throw "Docker failed to start llama.cpp. Error log tail:`n$errorTail"
        }

        throw "Docker failed to start llama.cpp with exit code $($process.ExitCode)."
    }

    if (Test-Path -LiteralPath $DockerRunOutLog) {
        $containerId = (Get-Content -LiteralPath $DockerRunOutLog -Raw).Trim()
        if ($containerId -ne "") {
            Write-Host $containerId
        }
    }
}

function Ensure-DockerImage {
    param([string] $Image)

    & docker image inspect $Image *> $null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "Pulling Docker image: $Image"
    & docker pull $Image | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Could not pull Docker image: $Image"
    }
}

function Test-PortListening {
    param([int] $Port)

    return (Get-PortOwningProcessId -Port $Port) -gt 0
}

function Get-PortOwningProcessId {
    param([int] $Port)

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $connection) {
            return 0
        }

        return [int] $connection.OwningProcess
    }
    catch {
        return 0
    }
}

function Test-HttpReady {
    param([string] $Uri)

    try {
        Invoke-RestMethod -Uri $Uri -TimeoutSec 5 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Wait-HttpReady {
    param(
        [string] $Name,
        [string] $Uri,
        [int] $TimeoutSeconds,
        [int] $ProcessId = 0,
        [string] $ErrorLogPath = ""
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpReady -Uri $Uri) {
            Write-Host "$Name is ready: $Uri"
            return
        }

        if ($ProcessId -gt 0 -and $null -eq (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            $errorTail = ""
            if ($ErrorLogPath -ne "" -and (Test-Path -LiteralPath $ErrorLogPath)) {
                $errorTail = (Get-Content -LiteralPath $ErrorLogPath -Tail 40 | Out-String).Trim()
            }

            if ($errorTail -ne "") {
                throw "$Name process exited before it became ready. Error log tail:`n$errorTail"
            }

            throw "$Name process exited before it became ready."
        }

        Write-Host "Waiting for $Name..."
        Start-Sleep -Seconds 5
    }

    throw "$Name did not become ready within $TimeoutSeconds seconds: $Uri"
}

function Stop-RecordedAppProcess {
    if (-not (Test-Path -LiteralPath $AppPidFile)) {
        return
    }

    $rawPid = [string] (Get-Content -LiteralPath $AppPidFile -Raw)
    $rawPid = $rawPid.Trim()
    if ($rawPid -eq "") {
        return
    }

    $processId = [int] $rawPid
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Host "Stopping existing app process $processId..."
        Stop-Process -Id $processId -Force
    }

    $portOwnerPid = Get-PortOwningProcessId -Port $AppPort
    if ($portOwnerPid -gt 0 -and $portOwnerPid -ne $processId) {
        $portOwner = Get-CimInstance Win32_Process -Filter "ProcessId = $portOwnerPid" `
            -ErrorAction SilentlyContinue
        if ($null -ne $portOwner -and $portOwner.CommandLine -like "*financial_research_agent*") {
            Write-Host "Stopping existing app port owner $portOwnerPid..."
            Stop-Process -Id $portOwnerPid -Force
        }
    }

    Remove-Item -LiteralPath $AppPidFile -ErrorAction SilentlyContinue
}

function Save-AppPortOwnerPid {
    $portOwnerPid = Get-PortOwningProcessId -Port $AppPort
    if ($portOwnerPid -gt 0) {
        Set-Content -LiteralPath $AppPidFile -Value $portOwnerPid -Encoding utf8
    }
}

function Get-PythonExecutable {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }

    if (Test-CommandAvailable -Name "python") {
        return (Get-Command python).Source
    }

    throw "Python was not found. Create .venv or install Python first."
}

function Test-PythonModule {
    param(
        [string] $PythonExe,
        [string] $ModuleName
    )

    & $PythonExe -c "import $ModuleName" *> $null
    return $LASTEXITCODE -eq 0
}

function Ensure-AppDependencies {
    param([string] $PythonExe)

    $missingModules = @()
    foreach ($moduleName in @("fastapi", "uvicorn", "httpx")) {
        if (-not (Test-PythonModule -PythonExe $PythonExe -ModuleName $moduleName)) {
            $missingModules += $moduleName
        }
    }

    if ($missingModules.Count -eq 0) {
        return
    }

    if ($SkipDependencyInstall) {
        throw "Missing Python modules: $($missingModules -join ', '). Run: $PythonExe -m pip install -e ."
    }

    Write-Host "Installing project runtime dependencies because these modules are missing: $($missingModules -join ', ')"
    & $PythonExe -m pip install -e $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency install failed. Run manually: $PythonExe -m pip install -e ."
    }
}

function Start-LlamaServer {
    if (-not (Test-CommandAvailable -Name "docker")) {
        throw "Docker was not found. Install/start Docker Desktop first."
    }

    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running. Start Docker Desktop first."
    }

    Ensure-DockerImage -Image $DockerImage
    New-Item -ItemType Directory -Force -Path $HuggingFaceCache | Out-Null

    if ($RestartLlama) {
        Write-Host "Restarting llama.cpp container '$ContainerName'..."
        if (-not (Remove-ContainerWithTimeout -Name $ContainerName)) {
            throw "Could not remove existing llama.cpp container '$ContainerName'. Restart Docker Desktop and try again."
        }
    }

    $isRunning = Get-ContainerRunning -Name $ContainerName
    if ($isRunning -eq $true) {
        $script:ActiveContainerName = $ContainerName
        Write-Host "llama.cpp container already running: $ContainerName"
        return
    }

    if ($isRunning -eq $false) {
        Write-Host "Removing stale llama.cpp container '$ContainerName'..."
        if (-not (Remove-ContainerWithTimeout -Name $ContainerName)) {
            $script:ActiveContainerName = New-FallbackContainerName -BaseName $ContainerName
            Write-Warning (
                "Could not remove stale container '$ContainerName' within 30 seconds. " +
                "Starting a new container named '$script:ActiveContainerName' instead."
            )
        }
    }

    if (Test-PortListening -Port $LlamaPort) {
        throw "Port $LlamaPort is already in use, but '$ContainerName' is not running."
    }

    Write-Host "Starting llama.cpp Docker server. First run may download the image and model."
    $dockerArgs = @(
        "run",
        "-d",
        "--rm",
        "--gpus",
        "all",
        "--name",
        $script:ActiveContainerName,
        "-p",
        "${LlamaPort}:8080",
        "-v",
        "${HuggingFaceCache}:/root/.cache/huggingface",
        $DockerImage,
        "-hf",
        $Model,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--ctx-size",
        "$ContextSize",
        "--jinja",
        "-ngl",
        "$GpuLayers"
    )

    Start-DockerRunWithTimeout -Arguments $dockerArgs
}

function Start-AppServer {
    param([string] $PythonExe)

    $statusUri = "http://${AppHost}:${AppPort}/api/status"
    if ($RestartApp) {
        Stop-RecordedAppProcess
    }
    elseif (Test-HttpReady -Uri $statusUri) {
        Write-Host "Chat UI already running: http://${AppHost}:${AppPort}"
        return 0
    }
    elseif (Test-PortListening -Port $AppPort) {
        throw "Port $AppPort is already in use, but the chat UI did not answer at $statusUri."
    }

    Write-Host "Starting Financial Research Agent chat UI..."
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Remove-Item -LiteralPath $AppOutLog, $AppErrLog -ErrorAction SilentlyContinue

    $previousEnv = @{
        FRA_LLM_PROVIDER = $env:FRA_LLM_PROVIDER
        FRA_LLM_MODEL = $env:FRA_LLM_MODEL
        FRA_LLM_BASE_URL = $env:FRA_LLM_BASE_URL
        FRA_LLM_LOCAL_RUNTIME = $env:FRA_LLM_LOCAL_RUNTIME
        PYTHONPATH = $env:PYTHONPATH
    }

    try {
        $env:FRA_LLM_PROVIDER = "local-openai"
        $env:FRA_LLM_MODEL = $Model
        $env:FRA_LLM_BASE_URL = "http://127.0.0.1:${LlamaPort}/v1"
        $env:FRA_LLM_LOCAL_RUNTIME = "llama.cpp"
        $env:PYTHONPATH = "$RepoRoot\src;$($env:PYTHONPATH)"

        $appProcess = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList @(
                "-m",
                "financial_research_agent",
                "serve",
                "--host",
                $AppHost,
                "--port",
                "$AppPort"
            ) `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $AppOutLog `
            -RedirectStandardError $AppErrLog `
            -PassThru

        Set-Content -LiteralPath $AppPidFile -Value $appProcess.Id -Encoding utf8
        Write-Host "Chat UI process started: $($appProcess.Id)"
        return $appProcess.Id
    }
    finally {
        foreach ($name in $previousEnv.Keys) {
            if ($null -eq $previousEnv[$name]) {
                Remove-Item -Path "env:$name" -ErrorAction SilentlyContinue
            }
            else {
                Set-Item -Path "env:$name" -Value $previousEnv[$name]
            }
        }
    }
}

$llamaModelsUri = "http://127.0.0.1:${LlamaPort}/v1/models"
$appStatusUri = "http://${AppHost}:${AppPort}/api/status"
$appUrl = "http://${AppHost}:${AppPort}"

Write-Host "Repository: $RepoRoot"
Write-Host "Model: $Model"
Write-Host "llama.cpp endpoint: http://127.0.0.1:${LlamaPort}/v1"
Write-Host "Chat UI: $appUrl"

$pythonExe = Get-PythonExecutable
Ensure-AppDependencies -PythonExe $pythonExe
Start-LlamaServer
$appProcessId = Start-AppServer -PythonExe $pythonExe

Wait-HttpReady -Name "llama.cpp" -Uri $llamaModelsUri -TimeoutSeconds $LlamaWaitSeconds
Wait-HttpReady `
    -Name "chat UI" `
    -Uri $appStatusUri `
    -TimeoutSeconds $AppWaitSeconds `
    -ProcessId $appProcessId `
    -ErrorLogPath $AppErrLog
Save-AppPortOwnerPid

if (-not $NoOpenBrowser) {
    Start-Process $appUrl
}

Write-Host ""
Write-Host "Ready."
Write-Host "Open: $appUrl"
Write-Host "App logs: $AppOutLog"
Write-Host "App errors: $AppErrLog"
Write-Host "Docker logs: docker logs -f $script:ActiveContainerName"
Write-Host ""
Write-Host "Stop later:"
Write-Host "  docker stop $script:ActiveContainerName"
if (Test-Path -LiteralPath $AppPidFile) {
    $recordedPid = (Get-Content -LiteralPath $AppPidFile -Raw).Trim()
    if ($recordedPid -ne "") {
        Write-Host "  Stop-Process -Id $recordedPid"
    }
}
