# Smartdelta ML Backend

CURL test commands for server:
```
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/train
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/anomaly_predict
curl -i -X POST -H "Content-Type: multipart/form-data" -F "userid=1" -F "filecomment=This is some CSV file" -F "file=@tst_file.csv" http://<server>:5003/next_hop_predict
```

