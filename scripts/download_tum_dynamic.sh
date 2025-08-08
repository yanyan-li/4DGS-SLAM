mkdir -p datasets/tum
cd datasets/tum


wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_xyz.tgz


wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_static.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_static.tgz
