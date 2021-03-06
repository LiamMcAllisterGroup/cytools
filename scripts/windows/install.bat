@echo off

docker info || echo. && echo "Please make sure that Docker is installed and running" && timeout /t 10 && exit

docker build -t cytools ../../ || echo. && echo "There was an error while building the image. Please let the developers know, and try using a stable version of the package." && timeout /t 10 && exit

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

%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUT1%\"); $S.TargetPath = \"%TARGET%\"; $S.IconLocation = \"%appdata%\CYTools\cytools.ico\"; $S.Save()"
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUT2%\"); $S.TargetPath = \"%TARGET%\"; $S.IconLocation = \"%appdata%\CYTools\cytools.ico\"; $S.Save()"
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut(\"%SHORTCUTU%\"); $S.TargetPath = \"%TARGETU%\"; $S.Save()"
