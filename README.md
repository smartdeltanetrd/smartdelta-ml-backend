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


# Categorical Comparison Route Documentation

This repository contains the backend API for SmartDelta Application's categorical comparison route.

#### Version: 0.0.1

#### Main technologies used:

`Python` `Flask` `Pandas` `NumPy` 


## API Usage

This section outlines how to use the `categorical_comparison` route.

### Request

Send a POST request to the `/categorical_comparison` route with two log files (CSV format) attached.

### Example Request using Axios (JavaScript):

Here's the request code part as a table in Markdown syntax:

```markdown
### Example Request using Axios (JavaScript)in Node.js:

```javascript
const formData = new FormData();

formData.append('file1', csvStream1, 'data1.csv');
formData.append('file2', csvStream2, 'data2.csv');

// Send POST request to the Flask API
const apiResponse = await axios.post('http://127.0.0.1:5006/categorical_comparison', formData, {
  headers: formData.getHeaders()
});
```

| Body Parameter | Type            | Description                                |
| -------------- | --------------- | ------------------------------------------ |
| `file1`        | `file/csv`      | **Requested**. CSV file for the first set of data. |
| `file2`        | `file/csv`      | **Requested**. CSV file for the second set of data. |

```

### Response

The API will respond with a JSON object containing the categorical comparison results.
Certainly! Here's the response code part as a table in Markdown syntax:


### Example Response JSON Object:

```json
{
  "file1": {
    "clusters": [
      {
        "message_instances": [
          {
            "fields": [
              {
                "Class": "com.genband.util.broker.util.MessageFactory"
              },
              {
                "Class.keyword": "com.genband.util.broker.util.MessageFactory"
              },
              // ... (other fields)
            ],
            "occurrence_percentage": 39.05
          },
          // ... (other instances)
        ],
        "occurrence_percentage": 85.71
      },
      // ... (other clusters)
    ]
  },
  "file2": {
    // ... (similar structure to file1)
  },
  "noise": [
    {
      "clusters1": [
        // ... (clusters from file1)
      ],
      "clusters2": [
        // ... (clusters from file2)
      ]
    }
  ]
}
```

| Key            | Value   | Description                                       |
| -------------- | ------- | ------------------------------------------------- |
| `file1`        | Object  | Results for the first CSV file.                   |
| `file2`        | Object  | Results for the second CSV file.                  |
| `noise`        | Array   | Information about noise instances in both files.  |

The response contains information about clusters, message instances, and noise instances for each file, along with their occurrence percentages.

## Example

Here's an example demonstrating how to use the `categorical_comparison` route in a Python script:

```python
import requests

# Load log files (csvStream1, csvStream2) ...

# Send request
url = 'http://127.0.0.1:5006/categorical_comparison'
files = {'file1': ('data1.csv', csvStream1), 'file2': ('data2.csv', csvStream2)}
response = requests.post(url, files=files)

# Print response
print(response.json())
```

Adjust the file loading and request parameters according to your specific implementation.

## Authors

![Logo](https://docs.kariyer.net/job/jobtemplate/000/000/241/avatar/24111520220128041051054.jpeg)
> Orion Innovation - 2021
