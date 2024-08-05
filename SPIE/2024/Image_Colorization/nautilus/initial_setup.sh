apt update && apt-get upgrade -y
apt install -y python3 python3-dev python3-pip
pip3 install --upgrade pip
apt install -y git wget unzip 
apt-get install -y vim tmux
apt install -y build-essential
apt-get install -y ffmpeg libsm6 libxext6

printf '2\n149' | wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh -b
cd /root/
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash