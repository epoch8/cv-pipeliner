# import json
# import requests
# from pathlib import Path

# examples_dir = Path(__file__).absolute().parent.parent / 'examples/brickit-ml/'
# tmp_dir = Path('tmp/')
# tmp_dir.mkdir(exist_ok=True)

# example_file = examples_dir / '35f20998-47ff-4d74-b2d9-245edc3dec32.jpeg'
# files = {'image': open(example_file, 'rb')}
# print(f'POST for {example_file}')
# response = requests.post("http://localhost:5000/predict", files=files)
# with open(tmp_dir / f'{example_file.stem}.json', 'w') as out:
#     json.dump(json.loads(response.text), out, indent=4)

# guid = 'AABBCC'
# response = requests.post(
#     f"http://localhost:5000/realtime_start/{guid}",
#     data={
#         'fps': 30,
#         'detection_delay': 300
#     }
# )
# frames_files = (examples_dir / 'easy').glob('*.png')
# for frame_file in frames_files:
#     print(f'POST for {frame_file}...')
#     response = requests.post(
#         f"http://localhost:5000/realtime_predict/{guid}",
#         files={'image': open(frame_file, 'rb')}
#     )
#     with open(tmp_dir / f'{frame_file.stem}.json', 'w') as out:
#         json.dump(json.loads(response.text), out, indent=4)
# response = requests.post(f"http://localhost:5000/realtime_end/{guid}")
