# Start from Debian Bullseye
FROM debian:bullseye

# Define build arguments
ARG USERNAME
ARG USERID
ARG ARCH
ARG AARCH
ARG VIRTUAL_ENV
ARG ALLOW_ROOT_ARG
ARG PORT_ARG
ENV ALLOW_ROOT=$ALLOW_ROOT_ARG
ENV PORT=$PORT_ARG

# Install dependencies
RUN apt-get -yqq update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -yqq install autoconf build-essential nano cmake libgmp-dev libcgal-dev\
                         libmpc-dev libsuitesparse-dev libppl-dev libeigen3-dev\
                         libc6 libcdd0d libgmp10 libgmpxx4ldbl libstdc++6 palp\
                         libflint-dev libflint-arb-dev python3 python3-pip curl\
                         wget libmath-libm-perl python3-venv normaliz libqsopt-ex2

# Make a soft link to the arb library and flint headers so that python-flint can install
RUN ln -s /usr/lib/${AARCH}-linux-gnu/libflint-arb.so /usr/lib/${AARCH}-linux-gnu/libarb.so
RUN ln -s /usr/include/flint/* /usr/include/

# Set up non-root user
RUN groupadd -r -g $USERID $USERNAME && useradd -r -s /bin/bash -u $USERID -g $USERNAME -m $USERNAME\
    || echo "Skipping user creation"
USER $USERNAME

# Install Rust since there are some Python packages that now depend on it
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/home/${USERNAME}/.cargo/bin:${PATH}"

# Create python virtual environment for non-root user
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip packages
ENV CVXOPT_SUITESPARSE_INC_DIR=/usr/include/suitesparse
WORKDIR /opt/cytools/
COPY ./requirements.txt /opt/cytools/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install python-flint==0.3.0
RUN pip3 install -f https://download.mosek.com/stable/wheel/index.html Mosek
ENV MOSEKLM_LICENSE_FILE=/home/$USERNAME/mounted_volume/mosek/mosek.lic

# Fix cvxopt bug
USER root
RUN sed -i -e 's/mosek.solsta.near_optimal/ /g' $VIRTUAL_ENV/lib/python3.9/site-packages/cvxopt/coneprog.py

# Install TOPCOM
WORKDIR /opt/cytools/external/topcom-mod
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v1.1.2%2Bds-1%2Bcytools-1/topcom_1.1.2+ds-1+cytools-1_${ARCH}.deb
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v1.1.2%2Bds-1%2Bcytools-1/libtopcom0_1.1.2+ds-1+cytools-1_${ARCH}.deb
RUN wget https://github.com/LiamMcAllisterGroup/topcom/releases/download/v1.1.2%2Bds-1%2Bcytools-1/libtopcom-dev_1.1.2+ds-1+cytools-1_${ARCH}.deb
RUN dpkg -i topcom_1.1.2+ds-1+cytools-1_${ARCH}.deb
RUN dpkg -i libtopcom0_1.1.2+ds-1+cytools-1_${ARCH}.deb
RUN dpkg -i libtopcom-dev_1.1.2+ds-1+cytools-1_${ARCH}.deb

# Download file from github to keep track of the number of downloads
RUN wget https://github.com/LiamMcAllisterGroup/cytools/releases/download/v1.0.0/download_counter.txt

# Copy code and installer
COPY . /opt/cytools/
WORKDIR /opt/cytools/
RUN python3 setup.py install

# Create CGAL code for different dimensions and compile
WORKDIR /opt/cytools/external/cgal
RUN for i in $(seq 2 5); do sed "27s/.*/typedef CGAL::Epick_d<CGAL::Dimension_tag<${i}> >    K;/" cgal-triangulate.cpp > "cgal-triangulate-${i}d.cpp"; done;

# Fix CGAL headers so that Eigen3 is imported correctly
RUN sed -i -e 's/Eigen\/Core/eigen3\/Eigen\/Core/g' /usr/include/CGAL/Dimension.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/LA.h
RUN sed -i -e 's/Eigen\/Dense/eigen3\/Eigen\/Dense/g' /usr/include/CGAL/NewKernel_d/LA_eigen/constructors.h

RUN cgal_create_CMakeLists
RUN sed -i -e 's/find_package/find_package( Eigen3 3.3 REQUIRED )\nfind_package/g' /opt/cytools/external/cgal/CMakeLists.txt
RUN cmake . -DCMAKE_BUILD_TYPE=Release
# Must be single-threaded or it crashes on macOS
RUN make -j 1
RUN for i in $(seq 2 5); do ln -s "/opt/cytools/external/cgal/cgal-triangulate-${i}d" "/usr/local/bin/cgal-triangulate-${i}d"; done
RUN ln -s "/opt/cytools/external/cgal/cgal-triangulate" "/usr/local/bin/cgal-triangulate"

# Set variables so that numpy is limited to one thread
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Set entry path
WORKDIR /home/$USERNAME/mounted_volume

# Start jupyter lab by default
USER $USERNAME
CMD jupyter lab --ip 0.0.0.0 --port $PORT --no-browser $ALLOW_ROOT
