# Start from Debian Bullseye
FROM debian:bullseye

ARG USERID
ARG ARCH
ARG AARCH

# Install dependencies
RUN apt-get -yqq update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -yqq install autoconf build-essential nano cmake libgmp-dev libcgal-dev\
                         libmpc-dev libsuitesparse-dev libppl-dev libeigen3-dev\
                         libc6 libcdd0d libgmp10 libgmpxx4ldbl libstdc++6 palp\
                         libflint-dev libflint-arb-dev python3 python3-pip\
                         cython3 wget libmath-libm-perl

# Make a soft link for python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python
# Make a soft link to the arb library so that python-flint can install
RUN echo testt ${AARCH}
RUN ln -s /usr/lib/${AARCH}-linux-gnu/libflint-arb.so /usr/lib/${AARCH}-linux-gnu/libarb.so

# Fix flint headers
RUN cp -r /usr/include/flint/* /usr/include/

# Set up non-root user
ARG USERID
RUN groupadd -r -g $USERID cytools && useradd -r -s /bin/bash -u $USERID -g cytools -m cytools
USER cytools

# Install pip packages
ENV CVXOPT_SUITESPARSE_INC_DIR=/usr/include/suitesparse
RUN pip3 install --upgrade pip
RUN pip3 install --no-warn-script-location numpy scipy jupyterlab cvxopt gekko pymongo ortools tqdm
RUN pip3 install --no-warn-script-location python-flint matplotlib
RUN pip3 install --no-warn-script-location --user scikit-sparse cysignals gmpy2==2.1.0a4
RUN pip3 install --no-warn-script-location pplpy
RUN pip3 install --no-warn-script-location -f https://download.mosek.com/stable/wheel/index.html Mosek
ENV MOSEKLM_LICENSE_FILE=/opt/cytools/external/mosek/mosek.lic

# Fix cvxopt bug
USER root
RUN sed -i -e 's/mosek.solsta.near_optimal/ /g' /home/cytools/.local/lib/python3.9/site-packages/cvxopt/coneprog.py

# Install TOPCOM
WORKDIR /opt/cytools/external/topcom-mod
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v0.17.8%2Bds-2%2Bcytools-1/topcom_0.17.8+ds-2+cytools-1_${ARCH}.deb
RUN dpkg -i topcom_0.17.8+ds-2+cytools-1_${ARCH}.deb

# Copy code and installer
COPY . /opt/cytools/
WORKDIR /opt/cytools/
RUN python3 setup.py install

# Change permissions of Mosek license in case there is a mismatch
RUN chmod 666 /opt/cytools/external/mosek/mosek.lic || echo "Missing Mosek license"

# Create CGAL code for different dimensions and compile
WORKDIR /opt/cytools/external/cgal
RUN for i in $(seq 1 6); do sed "26s/.*/const int D = ${i};/" triangulate.cpp > "triangulate-${i}d.cpp"; done; rm triangulate.cpp

# Fix CGAL headers so that Eigen3 is imported correctly
RUN sed -i -e 's/Eigen\/Core/eigen3\/Eigen\/Core/g' /usr/include/CGAL/Dimension.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/LA.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/constructors.h

RUN cgal_create_CMakeLists
RUN sed -i -e 's/find_package/find_package( Eigen3 3.3 REQUIRED )\nfind_package/g' /opt/cytools/external/cgal/CMakeLists.txt
RUN cmake . -DCMAKE_BUILD_TYPE=Release
# Must be single-threaded or it crashes on macOS
RUN make -j 1
RUN for i in $(seq 1 6); do ln -s "/opt/cytools/external/cgal/triangulate-${i}d" "/usr/local/bin/cgal-triangulate-${i}d"; done

# Set variables so that numpy is limited to one thread
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Set entry path
WORKDIR /home/cytools/mounted_volume

# Start jupyter lab by default
USER cytools
ENV PATH="/home/cytools/.local/bin:$PATH"
CMD jupyter lab --ip 0.0.0.0 --port 2875 --no-browser
