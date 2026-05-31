$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Install vcpkg packages
winget install -e --id Microsoft.VisualStudio.BuildTools --override "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools"
$script = New-TemporaryFile
Invoke-WebRequest https://aka.ms/vcpkg-init.ps1 -OutFile $script
& $script
Remove-Item $script
vcpkg install
$env:PATH = "C:\Users\egoud\naaulu\vcpkg_installed\x64-windows\bin;$env:PATH"
$env:NETCDF4_DIR = "C:\Users\egoud\naaulu\vcpkg_installed\x64-windows"

# Install python packages
winget install --id astral-sh.uv
uv python install 3.14t
uv venv --python 3.14t --no-clear
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]" --upgrade --no-binary h5py --no-binary netcdf4 --prerelease=allow --cache-dir C:\uvcache
deactivate

Write-Host ""
Write-Host "To activate the environment in future sessions run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "To exit it run: deactivate"
