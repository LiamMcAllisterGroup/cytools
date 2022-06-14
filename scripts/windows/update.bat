@echo off

cd %TMP%
set dirname=cytools-update-%RANDOM%
mkdir %dirname%
cd %dirname%
for /f "tokens=1,* delims=," %%A in ('curl -ks https://api.github.com/repos/LiamMcAllisterGroup/cytools/releases/latest ^| findstr "tarball_url"') do (
    echo '%%A'
    for /f "tokens=1,* delims=:" %%B in ('echo %%A') do (
        curl -kL -o cytools.tar.gz %%C
    )
    
)
tar -xz --strip-components 1 -f cytools.tar.gz
cd scripts\windows
install.bat
cd %TMP%
for /f %i in ('dir /a:d /s /b cytools-update-*') do rmdir /s /q %i
