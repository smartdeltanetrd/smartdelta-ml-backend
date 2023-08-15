# Smartdelta ML Backend

## Start server in intercative mode

1. Build and run container in `docker_container_src/xsession-home` directory, as described in its README
1. Go to the project directory
1. Setup Python virtual env
   1. If this is the first time, create the environment itself and setup requirements
      ```
      python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt
      ```
   1. If this environment is already present, just source it
      ```
      source .venv/bin/activate
      ```
1. Run server in interactive mode
   ```
   python3 server.py
   ```

## Start server daemon

1. Rebuild container for the daemon (from the project directory)
   ```
   docker build -f docker_container_src/server/Dockerfile --tag smartdelta_ml_server_daemon .
   ```
1. Start container
   ```
   docker run -d --restart always -p 5003:5003 --name smartdelta_ml_daemon smartdelta_ml_server_daemon
   ```
1. Ensure that container is up

## Test server

CURL test commands for server:
```
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/train
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/anomaly_predict
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/next_hop_predict
```

