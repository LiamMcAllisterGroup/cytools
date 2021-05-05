@echo off

set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NoProfile

%PWS% -File "%~p0%launcher.ps1"
