
# Make sure that Docker is installed and running
$dockerinfo = docker info
if ("$dockerinfo" -eq "" ) {
  Write-Host "`nError: Seems like Docker is not installed in your system."
  sleep 10
  exit
} elseif ( "$dockerinfo".Contains("error during connect") ) {
  Write-Host "`nError: The Docker daemon is not running. Please make sure it is running before starting CYTools."
  sleep 10
  exit
}

# Check if there is already a cytools container
$contid = docker ps -a -q -f "name=cytools"
if ( "$contid" -ne "" ) {
  $choices = "&Yes", "&No"
  $decision = $Host.UI.PromptForChoice("A CYTools container already exists!", "Do you want to remove it and start a new one?",$choices,1)
  if ($decision -eq 0) {
    Write-Host "`nStopping and removing old CYTools container..."
    $tmp = docker stop cytools
    # Check if we need to remove the stopped container
    $contid = docker ps -a -q -f "name=cytools"
    if ( "$contid" -ne "" ) {
      $tmp = docker rm cytools
    }
  }
}

$banner=@"

   █████████  █████ █████ ███████████                   ████
  ███░░░░░███░░███ ░░███ ░█░░░███░░░█                  ░░███
 ███     ░░░  ░░███ ███  ░   ░███  ░   ██████   ██████  ░███   █████
░███           ░░█████       ░███     ███░░███ ███░░███ ░███  ███░░
░███            ░░███        ░███    ░███ ░███░███ ░███ ░███ ░░█████
░░███     ███    ░███        ░███    ░███ ░███░███ ░███ ░███  ░░░░███
 ░░█████████     █████       █████   ░░██████ ░░██████  █████ ██████
  ░░░░░░░░░     ░░░░░       ░░░░░     ░░░░░░   ░░░░░░  ░░░░░ ░░░░░░

                Developed by Liam McAllister's Group
                  https://cy.tools | Version 1.4.3

"@

Write-Host $banner

# Initialize Docker container
Write-Host "Initializing CYTools container..."
$tmp = docker run --rm -d -it --name cytools -p 2875:2875 -v ${home}:/home/root/mounted_volume cytools
$contid = docker ps -a -q -f "name=cytools"
if ( "$contid" -eq "" ) {
  Write-Host "The Docker container failed to start. Please make sure that the CYTools Docker image has been built."
  sleep 10
  exit
}

# Wait for up to 30 seconds for Jupyter lab to initialize
$initialized = $false
for ($n = 1; $n -le 60; $n++) {
  sleep .5
  $logs = docker logs cytools
  if ("$logs".Contains("127.0.0.1")) {
    $initialized = $true
    break
  }
}

if (-not $initialized) {
  Write-Host "Something went wrong. Please make sure that the CYTools Docker image has been built."
  docker stop cytools
  sleep 10
  exit
}

foreach ($w in $logs.split(" ")) {
  if ("$w".Contains("127.0.0.1")) {
    $link = $w
    break
  }
}

# We open the link in a new browser tab
Start-Process "$link"
Write-Host "CYTools is now running. If a new tab in your browser was not opened, please copy and paste the following link into your web browser of choice."
Write-Host "$link"
Write-Host "******************************************************************"
Write-Host "*** To exit CYTools press Ctrl+C twice while on this terminal. ***"
Write-Host "******************************************************************`n"

docker attach cytools
