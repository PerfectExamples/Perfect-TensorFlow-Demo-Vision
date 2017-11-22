OSABR=$(echo $(uname)|tr '[:upper:]' '[:lower:]')
VERSION=1.4.0
DWN=/tmp/libtensorflow.tgz
URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-$OSABR-x86_64-$VERSION.tar.gz
echo $URL
wget $URL -O $DWN
tar xvf $DWN -C /usr/local ./lib/libtensorflow.so ./lib/libtensorflow_framework.so
rm -f $DWN
echo 'download AI model'
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -O /tmp/in.zip
echo 'unzip model files'
unzip /tmp/in.zip -d /tmp
swift build
