<!-- markdownlint-disable -->

<a href="../cv_pipeliner/inference_models/detection/yolov8.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `yolov8.py`






---

## <kbd>class</kbd> `YOLOv8_DetectionModel`




<a href="../cv_pipeliner/inference_models/detection/yolov8.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(model_spec: YOLOv8_ModelSpec)
```

YOLOv8 model initialization 



**Args:**
 
 - <b>`model_spec`</b> (YOLOv8_ModelSpec):  YOLOv8 Model specification 



**Raises:**
 
 - <b>`ValueError`</b>:  if passed wrong data type of model_spec 


---

#### <kbd>property</kbd> input_size





---

#### <kbd>property</kbd> model_spec







---

<a href="../cv_pipeliner/inference_models/detection/yolov8.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `predict`

```python
predict(
    input: List[ndarray],
    score_threshold: float,
    classification_top_n: int = None
) → Tuple[List[List[Tuple[int, int, int, int]]], List[List[Tuple[int, int]]], List[List[float]], List[List[str]], List[List[float]]]
```

Method to run model inference 



**Args:**
 
 - <b>`input`</b> (DetectionInput):  list of images 
 - <b>`score_threshold`</b> (float):  model confidence threshold 
 - <b>`classification_top_n`</b> (int, optional):  .... Defaults to None. 



**Returns:**
 
 - <b>`DetectionOutput`</b>:  List of boxes, keypoints, scores, classes 

---

<a href="../cv_pipeliner/inference_models/detection/yolov8.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `preprocess_input`

```python
preprocess_input(input: List[ndarray]) → List[ndarray]
```






---

## <kbd>class</kbd> `YOLOv8_ModelSpec`





---

#### <kbd>property</kbd> inference_model_cls










---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
