# Container to isolate environment for SmartDelta ML server

This container:
* inherits from Ubuntu container and also installs additional python modules from Poetry files
* installs graphic support and forwards X session by the following path: `local host => ssh login to remote server (with X forwarding) => launched container => remote server => local host`
* creates internal user in container to perform read/write operations under it (instead of "root")
* shares user home directory for read/write by internal user within container environment

**Build** command:  
```
docker build --network host -f Dockerfile --build-arg user_name=`id -nu` --build-arg group_name=`id -ng` --build-arg user_uid=`id -u` --build-arg user_gid=`id -g` --tag ${USER}_smartdelta_ml_server .
```

**Run** command:  
```
docker run --runtime=nvidia -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~:/home/${USER} --net=host ${USER}_smartdelta_ml_server bash
```

