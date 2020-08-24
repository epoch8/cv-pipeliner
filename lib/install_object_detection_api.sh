pkgs='protobuf-compiler'
if ! dpkg -s $pkgs >/dev/null 2>&1; then
  sudo apt-get install $pkgs -y
fi

DIRECTORY=models/
if [ -d "$DIRECTORY" ];
then
    cd models/
    git pull
    cd ..
else
    git clone https://github.com/tensorflow/models.git 
fi

cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip3 install .
