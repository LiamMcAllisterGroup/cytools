# Start from Debian Buster
FROM debian:buster

# Install dependencies
RUN apt-get -yqq update
RUN apt-get -yqq install build-essential nano cmake libgmp-dev libcgal-dev\
                         libmpc-dev libsuitesparse-dev libppl-dev libeigen3-dev\
                         libc6 libcdd0d libgmp10 libgmpxx4ldbl libstdc++6 palp\
                         python3 python3-pip sudo wget

# Make a soft link for python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# Fix CGAL headers so that Eigen3 is imported correctly
RUN sed -i -e 's/Eigen\/Core/eigen3\/Eigen\/Core/g' /usr/include/CGAL/Dimension.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/LA.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/constructors.h

# Create a non-root user
ENV SHELL=/bin/bash
RUN useradd -m cytools && echo "cytools:cytools" | chpasswd && adduser cytools sudo
USER cytools
ENV PATH="/home/cytools/.local/bin:${PATH}"

# Install pip packages
RUN python3 -m pip install --no-warn-script-location --upgrade pip
RUN python3 -m pip install --no-warn-script-location numpy scipy jupyterlab cvxopt gekko flint-py pymongo ortools tqdm
RUN python3 -m pip install --no-warn-script-location --user scikit-sparse cysignals gmpy2==2.1.0a4
RUN python3 -m pip install --no-warn-script-location pplpy
RUN python3 -m pip install --no-warn-script-location -f https://download.mosek.com/stable/wheel/index.html Mosek
ENV MOSEKLM_LICENSE_FILE=/opt/cytools/external/mosek/mosek.lic

USER root

# Fix cvxopt bug
RUN sed -i -e 's/mosek.solsta.near_optimal/ /g' /home/cytools/.local/lib/python3.7/site-packages/cvxopt/coneprog.py

# Install TOPCOM
WORKDIR /opt/cytools/external/topcom-mod
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v0.17.8%2Bds-2%2Bcytools-1/topcom_0.17.8+ds-2+cytools-1_amd64.deb
RUN dpkg -i topcom_0.17.8+ds-2+cytools-1_amd64.deb

# Copy code and installer
COPY . /opt/cytools/
WORKDIR /opt/cytools/
RUN python3 setup.py install

# Create CGAL code for different dimensions and compile
WORKDIR /opt/cytools/external/cgal
RUN for i in $(seq 1 6); do sed "26s/.*/const int D = ${i};/" triangulate.cpp > "triangulate-${i}d.cpp"; done; rm triangulate.cpp

RUN cgal_create_CMakeLists -c Eigen3
RUN cmake . -DCMAKE_BUILD_TYPE=Release
# Must be single-threaded or it crashes on macOS
RUN make -j 1
RUN for i in $(seq 1 6); do ln -s "/opt/cytools/external/cgal/triangulate-${i}d" "/usr/local/bin/cgal-triangulate-${i}d"; done

# Set variables so that numpy is limited to one thread
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

USER cytools

# Set entry path
WORKDIR /home/cytools/mounted_volume

# Start jupyter lab by default
CMD jupyter lab --ip 0.0.0.0 --port 2875 --no-browser
