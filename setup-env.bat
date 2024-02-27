::call conda remove --name video-label-tracker --all -y
call conda create -n video-label-tracker python=3.10 -y
call conda activate video-label-tracker
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
call pip install ultralytics
call pip install opencv-contrib-python
call pip install lapx==0.5.5
call pip install pyqt5
call conda deactivate
 