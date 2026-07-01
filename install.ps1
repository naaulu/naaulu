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

# Install uv
winget install --id astral-sh.uv

# Install Python and create virtual environment
uv python install 3.14t
uv venv --python 3.14t --no-clear
.\.venv\Scripts\Activate.ps1

# Install package with dev dependencies
uv pip install -e ".[dev]" --upgrade --no-binary h5py --no-binary netcdf4 --prerelease=allow

deactivate

Write-Host ""
Write-Host "Virtual environment created: .venv"
Write-Host "To activate it run: .\.venv\Scripts\Activate.ps1"
