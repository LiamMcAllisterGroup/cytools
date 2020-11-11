# Start from python3 base (in Debian Buster)
FROM python:3.7-buster

# Install dependencies
RUN apt-get -yqq update
RUN apt-get -yqq install build-essential nano cmake libgmp-dev libcgal-dev\
                         libmpc-dev libsuitesparse-dev libppl-dev libeigen3-dev\
                         libc6 libcdd0d libgmp10 libgmpxx4ldbl libstdc++6

# Fix CGAL headers so that Eigen3 is imported correctly
RUN sed -i -e 's/Eigen\/Core/eigen3\/Eigen\/Core/g' /usr/include/CGAL/Dimension.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/LA.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/constructors.h

# Install pip packages
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy jupyterlab cvxopt gekko flint-py pymongo ortools tqdm
RUN pip3 install --user scikit-sparse cysignals gmpy2==2.1.0a4
RUN pip3 install pplpy
RUN pip3 install -f https://download.mosek.com/stable/wheel/index.html Mosek
ENV MOSEKLM_LICENSE_FILE=/cytools-install/external/mosek/mosek.lic
# Fix cvxopt bug
RUN sed -i -e 's/mosek.solsta.near_optimal/ /g' /usr/local/lib/python3.7/site-packages/cvxopt/coneprog.py

# Copy code and installer
ADD . /cytools-install/
WORKDIR /cytools-install/
RUN python3 setup.py install
RUN mkdir temp

# Install TOPCOM
WORKDIR /cytools-install/external/topcom-mod
RUN dpkg -i topcom-0.17.8+ds-2+cytools-1.deb

# Compile CGAL code
WORKDIR /cytools-install/external/cgal
RUN cgal_create_CMakeLists -c Eigen3
RUN cmake . -DCMAKE_BUILD_TYPE=Release
RUN make
RUN ln -s /cytools-install/external/cgal/triangulate /usr/local/bin/cgal-triangulate

# Set entry path
WORKDIR /home/

# Set variables so that numpy is limited to one thread
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Start jupyter lab by default
CMD jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
