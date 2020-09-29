# Two-stage pipeliner
- Detection
- Classification

# App
How to build and run:

- Set the config `app/config.yaml`
```
make app-build
make app-run
```

# Backend
How to build and run:

- Set the config `backend/config.yaml` (TFLite models are supported only). Models can be downloaded from @bobokvsky in Slack.
```
make backend-build
make backend-run
```

POST: `http://localhost:5000/predict/<int:guid>`, required content_type: `image/jpg`.

Example of POST:
```
>>> import requests
>>> import json
>>> file = 'examples/brickit-ml/35f20998-47ff-4d74-b2d9-245edc3dec32.jpeg'
>>> headers = {'Content-type': 'image/jpg', 'Accept': 'application/json'}
>>> response = requests.post("http://localhost:5000/predict/0", headers=headers, data=open(file, 'rb'))
>>> print(json.dumps(json.loads(response.text), indent=4))

{
    "bboxes": [
        {
            "label": "92593",
            "xmax": 753,
            "xmin": 589,
            "ymax": 812,
            "ymin": 648
        },
        {
            "label": "92593",
            "xmax": 472,
            "xmin": 208,
            "ymax": 884,
            "ymin": 672
        },
        {
            "label": "32054",
            "xmax": 642,
            "xmin": 429,
            "ymax": 468,
            "ymin": 245
        },
        ...
    ],
    "guid": 0
}
```