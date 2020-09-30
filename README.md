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

POST: `http://localhost:5000/predict/`, required content_type: `image/jpg`.

Example of POST:
```
>>> import requests
>>> import json
>>> file = 'examples/brickit-ml/35f20998-47ff-4d74-b2d9-245edc3dec32.jpeg'
>>> headers = {'Content-type': 'image/jpg', 'Accept': 'application/json'}
>>> response = requests.post("http://localhost:5000/predict/", headers=headers, data=open(file, 'rb'))
>>> print(json.dumps(json.loads(response.text), indent=4))

{
    "bboxes": [
        {
            "label": "3437",
            "xmax": 753,
            "xmin": 589,
            "ymax": 812,
            "ymin": 648
        },
        {
            "label": "3011",
            "xmax": 472,
            "xmin": 208,
            "ymax": 884,
            "ymin": 672
        },
        {
            "label": "3001",
            "xmax": 642,
            "xmin": 429,
            "ymax": 468,
            "ymin": 245
        },
        ...
    ]
}
```

Realtime POST: you need personal `guid` for this:
1. POST `http://localhost:5000/realtime_start/<guid>`
2. For every frame, POST image to `http://localhost:5000/realtime_predict/<guid>`, required content_type: `image/jpg`.
3. POST `http://localhost:5000/realtime_end/<guid>`

Example of use is in `tests/backend_test.py`.
