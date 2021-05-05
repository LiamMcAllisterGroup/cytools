@echo off

rd /s /q "%appdata%\Microsoft\Windows\Start Menu\Programs\CYTools
del /q "%userprofile%\Desktop\CYTools.lnk"

docker info || echo. && echo "Please make sure that Docker is installed and running" && timeout /t 10 && exit

docker rmi cytools || echo. && echo "Failed to remove the image. It either doesn't exist or there is a container still running." && timeout /t 10 && exit

rd /s /q "%appdata%\CYTools"
