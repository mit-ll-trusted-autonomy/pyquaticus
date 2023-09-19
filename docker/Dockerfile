# syntax=docker/dockerfile:1
FROM ubuntu:jammy

RUN dpkg-reconfigure debconf --frontend=noninteractive
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y tzdata subversion python3-pip alacarte libncurses-dev g++ subversion cmake libfltk1.3-dev freeglut3-dev libpng-dev libjpeg-dev libxft-dev libxinerama-dev libtiff5-dev sudo x11-apps net-tools emacs-nox bc xterm net-tools iputils-ping rsync screen libcairo2-dev git vim && apt-get clean



RUN useradd -m -p "moos" moos && \
    usermod -a -G sudo moos && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER moos
WORKDIR /home/moos

RUN svn export https://oceanai.mit.edu/svn/moos-ivp-aro/trunk/ moos-ivp
RUN svn export https://oceanai.mit.edu/svn/moos-ivp-aquaticus-oai/trunk/ moos-ivp-aquaticus
RUN git clone https://github.com/westpoint-robotics/mdo-hurt-s.git
RUN git clone https://github.com/mit-ll-trusted-autonomy/pyquaticus.git


COPY moos-ivp-rlagent moos-ivp-rlagent
COPY scripts scripts
COPY gym-aquaticus gym-aquaticus
COPY logs logs
COPY missions_pyquaticus missions_pyquaticus
COPY requirements.txt requirements.txt

RUN sudo chown -R moos:moos moos-ivp-rlagent logs pyquaticus moos-ivp missions_pyquaticus scripts missions_pyquaticus requirements.txt mdo-hurt-s

ENV PATH=${PATH}:/home/moos/moos-ivp/bin:/home/moos/moos-ivp-aquaticus/bin:/home/moos/moos-ivp-rlagent/bin:/home/moos/pyquaticus/:/home/moos/scripts:/home/moos/mdo-hurt-s/moos-ivp-surveyor/bin \
IVP_BEHAVIOR_DIRS=/home/moos/moos-ivp/lib:/home/moos/moos-ivp-aquaticus/lib:/home/moos/moos-ivp-rlagent/lib \
PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3/dist-packages:/home/moos/.local/bin:/home/moos/pyquaticus \
SCRIPTS=/home/moos/scripts \
MOOSIVP_SOURCE_TREE_BASE=~/moos-ivp/

# Install base utilities
RUN sudo apt-get update \
    && sudo apt-get install -y build-essential \
    && sudo apt-get install -y wget \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*



RUN sudo pip3 install -r requirements.txt --find-links `pwd` --no-cache-dir

RUN cd moos-ivp && ./build.sh && cd ~
RUN cd moos-ivp-aquaticus && ./build.sh && cd ~
RUN cd moos-ivp-rlagent && ./build.sh && cd ~
RUN cd mdo-hurt-s/moos-ivp-surveyor && ./build.sh && cd

# These need to be installed separately in this order due to version conflicts
RUN pip install ray[rllib]==2.4.0
RUN pip install gymnasium==0.28.1

ENTRYPOINT ["/bin/bash/", "-c", "python3 ~/moos-ivp-rlagent/missions/oct_wp_competition-2022/pyquaticus_bridge_test.py"]



