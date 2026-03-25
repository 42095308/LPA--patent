param(
    [switch]$IncludeCache
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $projectRoot "outputs"
$cacheDir = Join-Path $projectRoot "data\cache"

function Ensure-Gitkeep {
    param([string]$DirPath)
    if (-not (Test-Path -LiteralPath $DirPath)) {
        New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
    }
    $gitkeep = Join-Path $DirPath ".gitkeep"
    if (-not (Test-Path -LiteralPath $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}

function Clear-DirPreserveGitkeep {
    param([string]$DirPath)
    if (-not (Test-Path -LiteralPath $DirPath)) {
        return
    }

    Get-ChildItem -LiteralPath $DirPath -Force |
        Where-Object { $_.Name -ne ".gitkeep" } |
        Remove-Item -Recurse -Force
}

Ensure-Gitkeep -DirPath $outputDir
Ensure-Gitkeep -DirPath $cacheDir

Clear-DirPreserveGitkeep -DirPath $outputDir
if ($IncludeCache) {
    Clear-DirPreserveGitkeep -DirPath $cacheDir
}

$legacyRootArtifacts = @(
    "simulation_results.json",
    "Z_crop.npy",
    "Z_crop_geo.npz",
    "Z_crop_meta.json",
    "ceiling.npy",
    "floor.npy",
    "layer_mid.npy",
    "graph_edges.npy",
    "graph_nodes.npy",
    "corridor_vis.png",
    "graph_vis.png",
    "huashan_final.png",
    "lpa_result.png",
    "path_vis.png",
    "path_cost_profile.png",
    "osm_human_risk_preview.png",
    "lpa_seed_sweep.csv",
    "lpa_seed_sweep_summary.json",
    "osm_feature_summary.json"
)

foreach ($name in $legacyRootArtifacts) {
    $path = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

Write-Host "Clean completed. outputs cleared; cache cleared: $($IncludeCache.IsPresent)."
