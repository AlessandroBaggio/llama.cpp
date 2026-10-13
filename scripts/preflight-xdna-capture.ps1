[CmdletBinding()]
param(
    [string]$BuildDir,
    [string]$XclbinPath = $env:XDNA_PREFLIGHT_XCLBIN,
    [string]$ProbePath
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$projectRoot = Split-Path (Split-Path $repo -Parent) -Parent
if (-not $ProbePath) { $ProbePath = Join-Path $repo 'build-xdna/bin/xdna-qwen-shape-baseline-probe.exe' }
if (-not $BuildDir) { $BuildDir = Join-Path $repo 'build-xdna' }
$report = [System.Collections.Generic.List[string]]::new()
function Stop-Preflight([string]$Message) { throw $Message }
function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { Stop-Preflight "Missing required file: $Path" }
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Resolve-Artifact([string]$BackendName, [string]$Override, [string]$FrozenRelative) {
    $backendPath = [Environment]::GetEnvironmentVariable($BackendName)
    if ($backendPath) { $path = $backendPath; $source = 'BACKEND_ENV' }
    elseif ($Override) { $path = $Override; $source = 'PREFLIGHT_OVERRIDE' }
    else { $path = $FrozenRelative; $source = 'FROZEN_DEFAULT' }
    $absolutePath = if ([IO.Path]::IsPathRooted($path)) { $path } else { Join-Path $projectRoot $path }
    [pscustomobject]@{ Source=$source; Path=[IO.Path]::GetFullPath($absolutePath) }
}
function Invoke-Build([string]$Target) {
    $cmd = "cmake --build `"$script:BuildDir`" --config Release --target $Target"
    Write-Host "BUILD: $cmd"
    & $script:cmake.Source --build $script:BuildDir --config Release --target $Target
    $code = $LASTEXITCODE
    Write-Host "BUILD_RESULT target=$Target exit=$code $(if ($code -eq 0) {'PASS'} else {'FAIL'})"
    if ($code -ne 0) { Stop-Preflight "Build failed at target $Target (exit $code); no hardware access." }
}
try {
    Push-Location $repo
    Write-Host '=== FREEZE ==='
    git status --short
    git rev-parse HEAD
    git branch --show-current
    $protected = 'ggml/src/ggml-cpu/ggml-cpu.c'
    $cpuHash = Get-Sha256 (Join-Path $repo $protected)
    if ($cpuHash -ne '503ae0f53e484ac8ef3a130dbec7c82f3abc72f0356a28d942d4f4a4bfd6fc2e') { Stop-Preflight "Protected ggml-cpu.c hash mismatch: $cpuHash" }
    foreach ($f in @('ggml/src/ggml-backend.cpp','ggml/src/ggml-xdna/ggml-xdna.cpp',$protected,'src/llama-context.cpp')) { Write-Host "SHA256 $f $(Get-Sha256 (Join-Path $repo $f))" }

    Write-Host '=== VS2022 ==='
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path $vswhere)) { Stop-Preflight "vswhere not found: $vswhere" }
    $vs = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $vs) { Stop-Preflight 'VS2022 with x64 C++ tools not found.' }
    $vs = $vs.Trim()
    $devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
    if (-not (Test-Path $devcmd)) { Stop-Preflight "VsDevCmd missing: $devcmd" }
    $envDump = & $env:ComSpec /d /c "call `"$devcmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) { Stop-Preflight 'VS2022 environment activation failed.' }
    foreach ($line in $envDump) { if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process') } }
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    $cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if (-not $cl -or -not $cmake) { Stop-Preflight 'cl.exe/cmake.exe unavailable after VS activation.' }
    Write-Host "VS2022 edition/path: $vs"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $clVersion = & $cl.Source /Bv 2>&1 | Select-String -Pattern 'Compiler Version' | Select-Object -First 1
    $ErrorActionPreference = $previousPreference
    if (-not $clVersion) { Stop-Preflight 'Could not determine MSVC compiler version.' }
    Write-Host "MSVC: $clVersion"
    Write-Host "CMake: $(& $cmake.Source --version | Select-Object -First 1)"
    $script:cmake = $cmake
    $script:BuildDir = (Resolve-Path $BuildDir).Path

    Write-Host '=== XRT / METADATA ==='
    $xrtRoot = 'C:\Xilinx\XRT'
    if (-not (Test-Path (Join-Path $xrtRoot 'include'))) { Stop-Preflight "XRT root unavailable: $xrtRoot" }
    $versionFiles = @(Get-ChildItem $xrtRoot -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'version|release' })
    $versionEvidence = $versionFiles | Select-String -Pattern '2\.21\.0' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $versionEvidence) { Stop-Preflight 'Could not prove installed XRT version is 2.21.0.' }
    $xrtVersion = (Select-String -Path (Join-Path $xrtRoot 'include/xrt/detail/version.h') -Pattern 'xrt_build_version\[\].*"([^"]+)"').Matches.Groups[1].Value
    if ($xrtVersion -ne '2.21.0') { Stop-Preflight "Unsupported XRT version: $xrtVersion" }
    Write-Host "XRT root: $xrtRoot; version: $xrtVersion"
    Write-Host "Python: $(python --version 2>&1)"
    Write-Host "MLIR-AIE: $($env:MLIR_AIE_PATH)"

    Write-Host '=== DIAGNOSTIC ENVIRONMENT ==='
    $allowedRuntimeEnv = @('GGML_XDNA_Q4K_RUNTIME_XCLBIN','GGML_XDNA_Q4K_M1024_INSTS','GGML_XDNA_Q4K_M4096_INSTS','GGML_XDNA_Q4K_M8192_INSTS','GGML_XDNA_Q4K_M9216_INSTS')
    $xdnaEnv = @(Get-ChildItem Env: | Where-Object Name -Like 'GGML_XDNA*')
    $forbiddenEnv = @($xdnaEnv | Where-Object Name -NotIn $allowedRuntimeEnv)
    Get-ChildItem Env: | Where-Object { $_.Name -Like 'XRT*' -or $_.Name -Like 'XILINX*' -or $_.Name -Like 'GGML_XDNA*' } | Sort-Object Name | Format-Table -AutoSize
    if ($forbiddenEnv) { $forbiddenEnv | Format-Table -AutoSize; Stop-Preflight 'FAIL ENV CLEANLINESS: diagnostic GGML_XDNA* variable present.' }
    if (Test-Path Env:BP1_USED_RECREATE) { Stop-Preflight 'FAIL ENV CLEANLINESS: BP1_USED_RECREATE present.' }
    Write-Host 'Diagnostic ENV: PASS (only production artifact-path variables are allowed)'
    $sourceFiles = @((Join-Path $repo 'ggml/src/ggml-xdna/ggml-xdna.cpp'),(Join-Path $repo 'ggml/src/ggml-backend.cpp'))
    $gates = Select-String -Path $sourceFiles -Pattern 'getenv\s*\(\s*"GGML_XDNA_[A-Z0-9_]+' | ForEach-Object {
        if ($_.Line -match '"(GGML_XDNA_[A-Z0-9_]+)"') { [pscustomobject]@{ Gate=$Matches[1]; Location="$($_.Path):$($_.LineNumber)"; Behavior=$_.Line.Trim(); DisabledNow='YES' } }
    } | Sort-Object Gate,Location -Unique
    $gates | Format-Table -Wrap -AutoSize

    Write-Host '=== ARTIFACT CONTRACT ==='
    $specs = @(
      @{ Name='XCLBIN'; Backend='GGML_XDNA_Q4K_RUNTIME_XCLBIN'; Override=$XclbinPath; Default='experiments/xdna-q4k-persistent-multichunk-ap1/ap1_128.xclbin'; Expected='b8d5bc1f77c7325696ee3c7c0496323b5e523fd296475560eb21c0884aea2d83' },
      @{ Name='M1024'; Backend='GGML_XDNA_Q4K_M1024_INSTS'; Override=$env:XDNA_PREFLIGHT_M1024_INSTS; Default='experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m1024.insts.bin'; Expected='58b01921a65b0d11374d4ca3db7d55d5355087be89d542d793125878e1ca2d7e' },
      @{ Name='M4096'; Backend='GGML_XDNA_Q4K_M4096_INSTS'; Override=$env:XDNA_PREFLIGHT_M4096_INSTS; Default='experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m4096.insts.bin'; Expected='0309cdff9e81f4ceb77933e291379da731be817a4033e7a60ba8322d594ed904' },
      @{ Name='M8192'; Backend='GGML_XDNA_Q4K_M8192_INSTS'; Override=$env:XDNA_PREFLIGHT_M8192_INSTS; Default='experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m8192.insts.bin'; Expected='72f88794f6dbb60f695840b8bec10726f1e2a34edfbb9f5e01079fd392950c8f' },
      @{ Name='M9216'; Backend='GGML_XDNA_Q4K_M9216_INSTS'; Override=$env:XDNA_PREFLIGHT_M9216_INSTS; Default='experiments/xdna-q4k-runtime-m-sweep-aq1/aq1_m9216.insts.bin'; Expected='3e0afc787e75aa706198fdf40f0cc96e5a7f9ed1d900fcdb6dd42e3499bb68d3' }
    )
    $artifactResults = foreach ($spec in $specs) {
        $resolved = Resolve-Artifact $spec.Backend $spec.Override $spec.Default
        $actual = Get-Sha256 $resolved.Path
        [pscustomobject]@{ Item=$spec.Name; Source=$resolved.Source; Path=$resolved.Path; SHA=$actual; Expected=$spec.Expected; Result=$(if ($actual -eq $spec.Expected) {'PASS'} else {'FAIL'}) }
    }
    $artifactResults | Format-Table -AutoSize -Wrap
    if (@($artifactResults | Where-Object Result -ne 'PASS').Count -gt 0) { Stop-Preflight 'Frozen artifact SHA contract failed; no build or hardware access.' }
    foreach ($item in $artifactResults) { $spec = $specs | Where-Object Name -eq $item.Item; [Environment]::SetEnvironmentVariable($spec.Backend,$item.Path,'Process') }
    Write-Host 'ARTIFACT CONTRACT: PASS'

    Write-Host '=== BUILD ==='
    foreach ($target in @('ggml-base','ggml-xdna','llama-cli')) { Invoke-Build $target }
    $probeSource = Join-Path $repo 'build-xdna/xdna-qwen-shape-baseline-probe.cpp'
    if (-not (Test-Path -LiteralPath $ProbePath -PathType Leaf) -or -not (Test-Path -LiteralPath $probeSource -PathType Leaf)) { Stop-Preflight 'Known creation-only clean-state probe source/binary unavailable.' }
    $probeText = Get-Content -LiteralPath $probeSource -Raw
    if ($probeText -notmatch 'ggml_backend_dev_init' -or $probeText -notmatch 'ggml_backend_free' -or $probeText -match 'graph_compute|run\.wait|kernel\s*\(') { Stop-Preflight 'Probe violates initialization/teardown-only contract.' }
    $probeOut = [IO.Path]::GetFullPath($ProbePath)
    $probeLibDir = Join-Path $BuildDir 'ggml/src'
    $probeCommand = "cl.exe /nologo /EHsc /MD /I`"$(Join-Path $repo 'ggml/include')`" `"$probeSource`" /link /LIBPATH:`"$probeLibDir`" ggml-base.lib ggml.lib /OUT:`"$probeOut`""
    Write-Host "BUILD: $probeCommand"
    & $cl.Source /nologo /EHsc /MD "/I$(Join-Path $repo 'ggml/include')" "/Fo$(Join-Path $BuildDir 'xdna-qwen-shape-baseline-probe.obj')" $probeSource /link "/LIBPATH:$probeLibDir" ggml-base.lib ggml.lib "/OUT:$probeOut"
    $probeBuildCode = $LASTEXITCODE
    Write-Host "BUILD_RESULT target=clean-state-probe exit=$probeBuildCode $(if ($probeBuildCode -eq 0) {'PASS'} else {'FAIL'})"
    if ($probeBuildCode -ne 0) { Stop-Preflight 'Clean-state probe build failed; no hardware access.' }
    $cpuAfterBuild = Get-Sha256 (Join-Path $repo $protected)
    if ($cpuAfterBuild -ne $cpuHash) { Stop-Preflight 'Protected ggml-cpu.c changed during build.' }
    $probeArg = "profile=q4k_q8k_k2560;$($artifactResults | Where-Object Item -eq 'XCLBIN' | Select-Object -ExpandProperty Path)"
    Push-Location (Split-Path $probeOut -Parent)
    try {
        foreach ($gate in 1,2) {
            Write-Host "=== CLEAN-STATE GATE #$gate (fresh process) ==="
            $gatePreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            $gateOutput = & $probeOut $probeArg 2>&1
            $ErrorActionPreference = $gatePreference
            $gateCode = $LASTEXITCODE
            $gateOutput | ForEach-Object { Write-Host $_ }
            $joinedOutput = $gateOutput -join "`n"
            foreach ($marker in @('PRIMARY READY','M1024 READY','M4096 READY','M8192 READY','M9216 READY','kernel submissions=0','BACKEND FREE PASS','CLEAN-STATE PASS')) {
                if ($joinedOutput -notmatch [regex]::Escape($marker)) { Stop-Preflight "Clean-state gate #$gate missing marker: $marker" }
            }
            if ($gateCode -ne 0) { Stop-Preflight "Clean-state gate #$gate failed (exit $gateCode). No further XRT process will run." }
            Write-Host "CLEAN_GATE_$gate PASS; process exit=$gateCode; kernel submissions=0"
        }
    } finally { Pop-Location }
    $cpuAtExit = Get-Sha256 (Join-Path $repo $protected)
    if ($cpuAtExit -ne $cpuHash) { Stop-Preflight 'Protected ggml-cpu.c changed during preflight.' }
    Write-Host 'XDNA capture readiness: capture incomplete; exact-history replay missing (no capture replay execution).'
} catch {
    Write-Error "PREFLIGHT FAIL: $($_.Exception.Message)"
    exit 1
} finally {
    Pop-Location -ErrorAction SilentlyContinue
}
