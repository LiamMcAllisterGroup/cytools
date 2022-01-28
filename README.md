
<p align="center">
    <img src="https://cytools.liammcallistergroup.com/img/titleimage-circle.svg?sanitize=true" height="250"></img><br></br>
    <b>A software package for analyzing Calabi-Yau hypersurfaces in toric varieties.</b><br></br>
    <img alt="Latest release" src="https://img.shields.io/github/v/release/liammcallistergroup/cytools"></img>
    <img alt="Number of downloads" src="https://img.shields.io/github/downloads/liammcallistergroup/topcom/total"></img>
    <img alt="Lines of code" src="https://img.shields.io/tokei/lines/github/liammcallistergroup/cytools"></img>
    <img alt="License" src="https://img.shields.io/github/license/liammcallistergroup/cytools"></img>
</p>

-------------------------------------------------------------------------------

### Important: This package is currently in beta. If you want to be a beta tester please contact Mehmet Demirtas (&#109;&#46;&#100;&#101;&#109;&#105;&#114;&#116;&#97;&#115;&#64;&#110;&#111;&#114;&#116;&#104;&#101;&#97;&#115;&#116;&#101;&#114;&#110;&#46;&#101;&#100;&#117;) or Andres Rios-Tascon (&#97;&#114;&#50;&#50;&#56;&#53;&#64;&#99;&#111;&#114;&#110;&#101;&#108;&#108;&#46;&#101;&#100;&#117;).

CYTools is an open-source software package developed by [Liam McAllister's group](https://liammcallistergroup.com/) with the purpose of studying Calabi-Yau manifolds arising from the Kreuzer-Skarke database. It emerged from several years of effort towards exploring previously uncharted parts of the string landscape. It offers vastly superior computational performance compared to other software that are typically used in the field. Installation instructions and detailed documentation can be found in the [CYTools website](https://cytools.liammcallistergroup.com).

Most of the code is written in Python, with wrappers to interface with various other open-source software. The full package comes in the form of a Docker image that includes all the necessary dependencies, so that it is easy to install and use on most operating systems.

CYTools makes use a variety of open-source projects. It includes a few code snippets from [SageMath](https://www.sagemath.org/) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)], a modified version of [TOPCOM](http://www.rambau.wm.uni-bayreuth.de/TOPCOM/) [[GPLv2](http://www.gnu.org/licenses/gpl-2.0.html)] that can be found [here](https://github.com/LiamMcAllisterGroup/topcom), the [Computational Geometry Algorithms Library](https://www.cgal.org) [[LGPLv3](http://www.gnu.org/licenses/lgpl-3.0.html)], and multiple Python packages including [SciPy](https://www.scipy.org/), [NumPy](https://numpy.org/), [pplpy](https://gitlab.com/videlec/pplpy), [OR-Tools](https://developers.google.com/optimization), [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), and [flint-py](https://gitlab.com/alisianoi/flint-py).

All original CYTools code is distributed under the terms of the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt). All other packages and code snippets are redistributed under their respective licenses.

The current lead developers are Mehmet Demirtas (&#109;&#46;&#100;&#101;&#109;&#105;&#114;&#116;&#97;&#115;&#64;&#110;&#111;&#114;&#116;&#104;&#101;&#97;&#115;&#116;&#101;&#114;&#110;&#46;&#101;&#100;&#117;) and Andres Rios-Tascon (&#97;&#114;&#50;&#50;&#56;&#53;&#64;&#99;&#111;&#114;&#110;&#101;&#108;&#108;&#46;&#101;&#100;&#117;). Questions, comments and/or suggestions can be directed to either of us.
