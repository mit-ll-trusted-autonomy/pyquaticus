mkdir -p ./logs
docker build -t pyaquaticus:v1 . &&
xhost local:docker &&
docker run -ti --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=!" 20230712_docker_oct_entries_test:v1

