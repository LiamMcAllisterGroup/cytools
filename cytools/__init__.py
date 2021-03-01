# Make the main classes and function accessible from the root of CYTools.
from cytools.polytope import Polytope
from cytools.cone import Cone
from cytools.utils import read_polytopes, fetch_polytopes

# Latest version
version = "0.0.1"
versions_with_serious_bugs = []

# Check for more recent versions of CYTools
def check_for_updates():
    import requests
    try:
        p = requests.get("https://raw.githubusercontent.com/LiamMcAllisterGroup/cytools/main/cytools/__init__.py",
                         timeout=2)
        for l in p.text.split("\n"):
            if "version"+" =" in l:
                latest_ver = tuple(int(c) for c in l.split("\"")[1].split("."))
                ver = tuple(int(c) for c in version.split("."))
                if latest_ver <= ver:
                    continue
                print("Info: A more recent version of CYTools is available. "
                      "We recommend upgrading before continuing.")
            elif "versions_with_serious_bugs"+" =" in l:
                bad_versions = literal_eval(l.split("=")[1])
                if version in bad_versions:
                    print("****************************\n"
                          "Warning: This version of CYTools contains a serious"
                          "bug. Please upgrade to the latest version.\n"
                          "****************************\n")
    except:
       pass
check_for_updates()
