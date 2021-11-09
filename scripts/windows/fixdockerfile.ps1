# We replace a few lines in the Dockerfile so that the default user is root

(Get-content ..\..\Dockerfile) | %{$_ -replace "USER cytools","# Line removed for Windows Dockerfile"} | Set-content ..\..\Dockerfile
(Get-content ..\..\Dockerfile) | %{$_ -replace "/home/cytools/.local/lib/python3.7/site-packages","/usr/local/lib/python3.7/dist-packages"} | Set-content ..\..\Dockerfile