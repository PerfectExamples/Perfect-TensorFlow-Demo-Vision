OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
DWN=/tmp/libtensorflow.tgz
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-1.2.0.tar.gz
echo $URL
wget $URL -O $DWN
sudo tar xvf $DWN -C /usr/local ./lib/libtensorflow.so
rm -f $DWN
echo 'download AI model'
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O /tmp/in.zip
echo 'unzip model files'
unzip /tmp/in.zip -d /tmp
swift build
