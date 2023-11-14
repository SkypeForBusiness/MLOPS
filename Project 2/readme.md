# MLOPS Project 2: Docker

## Build image yourself
1. Install docker desktop https://www.docker.com/products/docker-desktop/
2. Download this Repository
3. Open a terminal and navigate to the download location
4. Run this command ```docker build -t mlops-project2 .```
5. Add a empty folder for model checkpoints in /src
6. Once the image is built run ```docker run -it mlops-project2 python /code/src/train.py --checkpoint_dir models``` to start the container

## Docker Image from Dockerhub on Play with Docker
1. Start play with docker https://labs.play-with-docker.com/#
2. Pull docker image ```docker pull skypeforbusiness/mlops-project2```
3. Run the training script ```docker run -it skypeforbusiness/mlops-project2 python /code/src/train.py --checkpoint_dir models --wandb disabled --batch_size 5```
