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

# Install TOPCOM
WORKDIR /cytools-install/external/topcom-mod
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v0.17.8%2Bds-2%2Bcytools-1/topcom_0.17.8+ds-2+cytools-1_amd64.deb
RUN dpkg -i topcom_0.17.8+ds-2+cytools-1_amd64.deb

# Install PALP
WORKDIR /cytools-install/external/
RUN wget http://hep.itp.tuwien.ac.at/~kreuzer/CY/palp/palp-2.20.tar.gz
RUN tar xvf palp-2.20.tar.gz; rm palp-2.20.tar.gz; mv palp-2.20 palp
WORKDIR /cytools-install/external/palp
RUN make

# Copy code and installer
ADD . /cytools-install/
WORKDIR /cytools-install/
RUN python3 setup.py install
RUN mkdir temp

# Create CGAL code for different dimensions and compile
WORKDIR /cytools-install/external/cgal
RUN for i in $(seq 1 10); do sed "26s/.*/const int D = ${i};/" triangulate.cpp > "triangulate-${i}d.cpp"; done; rm triangulate.cpp

RUN cgal_create_CMakeLists -c Eigen3
RUN cmake . -DCMAKE_BUILD_TYPE=Release
# Must be single-threaded or it crashes on macOS
RUN make -j 1
RUN for i in $(seq 1 10); do ln -s "/cytools-install/external/cgal/triangulate-${i}d" "/usr/local/bin/cgal-triangulate-${i}d"; done

# Set entry path
WORKDIR /home/

# Set variables so that numpy is limited to one thread
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Start jupyter lab by default
CMD jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
