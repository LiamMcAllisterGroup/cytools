@echo off

set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NoProfile

docker info || echo. && echo "Please make sure that Docker is installed and running" && timeout /t 10 && exit

docker rmi cytools || echo "Old CYTools image does not exist or cannot be deleted"
docker build --no-cache --force-rm -t cytools --build-arg USERNAME=root --build-arg USERID=0 --build-arg ARCH=amd64 --build-arg AARCH=x86_64 --build-arg VIRTUAL_ENV=/opt/cytools/cytools-venv/ --build-arg ALLOW_ROOT_ARG="--allow-root" --build-arg PORT_ARG=2875 ../../ || echo. && echo "There was an error while building the image. Please let the developers know, and try using a stable version of the package." && timeout /t 10 && exit

set TARGET="%appdata%\CYTools\cytools.bat"
set TARGETU="%appdata%\CYTools\uninstall.bat"
set SHORTCUT1="%userprofile%\Desktop\CYTools.lnk"
set SHORTCUT2="%appdata%\Microsoft\Windows\Start Menu\Programs\CYTools\CYTools.lnk"
set SHORTCUTU="%appdata%\Microsoft\Windows\Start Menu\Programs\CYTools\Uninstall.lnk"
set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive -NoProfile

mkdir "%appdata%\CYTools"
copy cytools.bat "%appdata%\CYTools\cytools.bat"
copy launcher.ps1 "%appdata%\CYTools\launcher.ps1"
copy uninstall.bat "%appdata%\CYTools\uninstall.bat"
copy cytools.ico "%appdata%\CYTools\cytools.ico"

mkdir "%appdata%\Microsoft\Windows\Start Menu\Programs\CYTools"

mkdir "%userprofile%\mosek" || echo "Mosek directory already exists"
copy ..\..\external\mosek\mosek.lic "%userprofile%\mosek\mosek.lic" || echo "New Mosek license was not found"

%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUT1%\"); $S.TargetPath = \"%TARGET%\"; $S.IconLocation = \"%appdata%\CYTools\cytools.ico\"; $S.Save()"
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUT2%\"); $S.TargetPath = \"%TARGET%\"; $S.IconLocation = \"%appdata%\CYTools\cytools.ico\"; $S.Save()"
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUTU%\"); $S.TargetPath = \"%TARGETU%\"; $S.Save()"
