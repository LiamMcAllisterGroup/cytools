import sys
import numpy as np

output_path = "./docs/documentation/"

hline = "-------------------------------------------------------------------------------\n"

# All the files that are associated with a class should be listed first
filenames = ["polytope", "polytopeface", "triangulation", "toricvariety", "calabiyau", "cone", "utils", "config", "__init__"]
classnames = ["Polytope", "PolytopeFace", "Triangulation", "ToricVariety", "CalabiYau", "Cone"]

classdocstrings = []
classfunnames = [[] for i in range(len(classnames))]
classfundocstrings = [[] for i in range(len(classnames))]
miscfunnames = [[] for i in range(len(filenames))]
miscfundocstrings = [[] for i in range(len(filenames))]

def main(in_dir, out_dir):
    for n in range(len(filenames)):
        f = open(in_dir+f"{filenames[n]}.py", "r")
        lines = list(f.readlines())
        f.close()
        in_class = False
        class_id = -1
        l = 0
        while l < len(lines):
            ll = lines[l]
            for nn in range(len(classnames)):
                # Check if class definition starts
                if f"class {classnames[nn]}:" in ll:
                    in_class = True
                    class_id = nn
                    l += 1
                    ll = lines[l]
                    # Check if class has docstring
                    if "\"\"\"" in ll:
                        docstring = ""
                        is_one_line = False
                        example_section = ""
                        started_example_section = False
                        if len(ll) > 8:
                            docstring += ll[4:-4]
                            is_one_line = True
                        while not is_one_line:
                            l += 1
                            ll = lines[l]
                            if "\"\"\"" in ll:
                                docstring += ll[4:-4]
                                if started_example_section:
                                    example_section += "\n</details>\n"
                                    docstring += example_section
                                else:
                                    print(f"Warning: class {classnames[nn]} doesn't have example")
                                break
                            elif "**Example:**" in ll:
                                started_example_section = True
                                example_section = "\n<details><summary><b>Example</b></summary>\n\n"
                            elif started_example_section:
                                example_section += ll[4*(len(ll)>1):]
                            else:
                                docstring += ll[4*(len(ll)>1):]
                        classdocstrings.append(docstring)
                    break
            if in_class and len(ll) > 2 and ll[:4] != "    ":
                in_class = False
                class_id = -1
            if "def " in ll:
                fun_name = ll[4+4*in_class:ll.find("(")]
                docstring = ""
                has_docstring = False
                while True:
                    if "):" in ll or ll[-2:] == ":\n":
                        break
                    l += 1
                    ll = lines[l]
                l += 1
                ll = lines[l]
                if "\"\"\"" in ll:
                    is_one_line = False
                    example_section = ""
                    started_example_section = False
                    if len(ll) > 8+4*in_class:
                        docstring += ll[4+4*in_class:-4]
                        is_one_line = True
                    while not is_one_line:
                        l += 1
                        ll = lines[l]
                        if "\"\"\"" in ll:
                            docstring += ll[4+4*in_class:-4]
                            if started_example_section:
                                example_section += "\n</details>\n"
                                docstring += example_section
                            else:
                                print(f"Warning: function {fun_name} doesn't have example")
                            break
                        elif "**Example:**" in ll:
                            started_example_section = True
                            example_section = "\n<details><summary><b>Example</b></summary>\n\n"
                        elif started_example_section:
                            example_section += ll[(4+4*in_class)*(len(ll)>1):]
                        else:
                            docstring += ll[(4+4*in_class)*(len(ll)>1):]
                    (classfunnames if class_id != -1 else miscfunnames)[n].append(fun_name)
                    (classfundocstrings if class_id != -1 else miscfundocstrings)[n].append(docstring)
            l += 1
    # Save files with documentation
    # First start with the file for each class
    for n in range(len(classnames)):
        f = open(f"{out_dir}{filenames[n]}.md", "w")
        f.write(f"---\nid: {filenames[n]}\ntitle: {classnames[n]} Class\n---\n")
        f.write(f"{classdocstrings[n]}\n")
        f.write(f"{hline}\n")
        sorted_ind = np.argsort(classfunnames[n])
        # Frist write visible functions
        f.write(f"## Functions\n\n")
        for nn in range(len(sorted_ind)):
            if classfunnames[n][sorted_ind[nn]][0] == "_":
                continue
            f.write(f"### ```{classfunnames[n][sorted_ind[nn]]}```\n\n")
            f.write(f"{classfundocstrings[n][sorted_ind[nn]]}\n")
            f.write(f"{hline}\n")
        # Then write hidden functions
        f.write(f"## Hidden Functions\n\n")
        for nn in range(len(sorted_ind)):
            if classfunnames[n][sorted_ind[nn]][0] != "_":
                continue
            f.write(f"### ```{classfunnames[n][sorted_ind[nn]]}```\n\n")
            f.write(f"{classfundocstrings[n][sorted_ind[nn]]}\n")
            f.write(f"{hline}\n")
        f.close()
    # Finally, write miscellaneous functions
    f = open(f"{out_dir}other.md", "w")
    f.write("---\nid: other\ntitle: Miscellaneous Functions\nsidebar_label: Misc Functions\n---\n\n")
    f.write(f"There are various other functions in CYTools that don't belong to any particular class. They are defined in different places according to where they most closely belong. Here we list the location of the definitions of these functions as well as their documentation.\n")
    for n in range(len(filenames)):
        if len(miscfunnames[n]) == 0:
            continue
        f.write(f"## Functions in ```cytools.{filenames[n]}```\n\n")
        sorted_ind = np.argsort(miscfunnames[n])
        for nn in range(len(miscfunnames[n])):
            f.write(f"### ```{miscfunnames[n][sorted_ind[nn]]}```\n\n")
            f.write(f"{miscfundocstrings[n][sorted_ind[nn]]}\n")
            f.write(f"{hline}\n")
    f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Source code location must be specified")
    in_dir = sys.argv[1]
    out_dir = (output_path if len(sys.argv) == 2 else sys.argv[2])
    main(in_dir, out_dir)
