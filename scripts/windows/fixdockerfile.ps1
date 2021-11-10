# We replace a few lines in the Dockerfile so that the default user is root

(Get-content ..\..\Dockerfile) | %{$_ -replace "USER cytools","# Line removed for Windows Dockerfile"} | Set-content ..\..\Dockerfile
(Get-content ..\..\Dockerfile) | %{$_ -replace "/home/cytools/.local/lib/python3.7/site-packages","/usr/local/lib/python3.7/dist-packages"} | Set-content ..\..\Dockerfile
(Get-content ..\Dockerfile) | %{$_ -replace "CMD jupyter lab --ip 0.0.0.0 --port 2875 --no-browser","CMD jupyter lab --ip 0.0.0.0 --port 2875 --allow-root --no-browser"} | Set-content ..\Dockerfile
