from pathlib import Path
from io import BytesIO

import imageio
import dataframe_image as dfi
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_images_data_side_by_side

from cv_pipeliner.metrics.classification import get_df_classification_metrics
from cv_pipeliner.utils.images import concat_images

test_dir = Path(__file__).parent / "images"
test_dir.mkdir(exist_ok=True, parents=True)
image_path = Path(__file__).parent / "A_B_Z.jpg"

true_bbox_data_A1 = BboxData(image_path=image_path, xmin=131, ymin=116, xmax=396, ymax=373, label="A")
pred_bbox_data_A1 = BboxData(
    image_path=image_path,
    xmin=169,
    ymin=149,
    xmax=355,
    ymax=341,
    label="A",
    top_n=4,
    labels_top_n=["A", "B", "Z", "other"],
    classification_scores_top_n=[0.3, 0.5, 0.1, 0.1],
)

true_bbox_data_A2 = BboxData(image_path=image_path, xmin=64, ymin=656, xmax=321, ymax=900, label="A")
pred_bbox_data_B2 = BboxData(
    image_path=image_path,
    xmin=115,
    ymin=705,
    xmax=273,
    ymax=855,
    label="B",
    top_n=4,
    labels_top_n=["B", "A", "Z", "other"],
    classification_scores_top_n=[0.5, 0.1, 0.1, 0.3],
)

true_bbox_data_A3 = BboxData(image_path=image_path, xmin=625, ymin=75, xmax=877, ymax=329, label="A")
pred_bbox_data_Z3 = BboxData(
    image_path=image_path,
    xmin=671,
    ymin=119,
    xmax=837,
    ymax=289,
    label="Z",
    top_n=4,
    labels_top_n=["Z", "B", "other", "A"],
    classification_scores_top_n=[0.6, 0.3, 0.05, 0.05],
)

true_bbox_data_B4 = BboxData(image_path=image_path, xmin=581, ymin=429, xmax=921, ymax=767, label="B")
pred_bbox_data_B4 = BboxData(
    image_path=image_path,
    xmin=636,
    ymin=487,
    xmax=859,
    ymax=709,
    label="B",
    top_n=4,
    labels_top_n=["B", "Z", "A", "other"],
    classification_scores_top_n=[0.25, 0.25, 0.25, 0.25],
)
true_bbox_data_other6 = BboxData(image_path=image_path, xmin=356, ymin=709, xmax=567, ymax=937, label="Z")
pred_bbox_data_B6 = BboxData(
    image_path=image_path,
    xmin=393,
    ymin=757,
    xmax=533,
    ymax=900,
    label="B",
    top_n=4,
    labels_top_n=["B", "A", "Z", "other"],
    classification_scores_top_n=[0.3, 0.2, 0.3, 0.2],
)

true_bbox_data_B7 = BboxData(image_path=image_path, xmin=689, ymin=789, xmax=885, ymax=980, label="B")
pred_bbox_data_other7 = BboxData(
    image_path=image_path,
    xmin=724,
    ymin=821,
    xmax=846,
    ymax=952,
    label="other",
    top_n=4,
    labels_top_n=["other", "B", "A", "Z"],
    classification_scores_top_n=[0.4, 0.3, 0.1, 0.2],
)

true_bbox_data_other8 = BboxData(image_path=image_path, xmin=49, ymin=416, xmax=258, ymax=636, label="A")
pred_bbox_data_A8 = BboxData(
    image_path=image_path,
    xmin=81,
    ymin=452,
    xmax=223,
    ymax=592,
    label="other",
    top_n=4,
    labels_top_n=["other", "A", "B", "Z"],
    classification_scores_top_n=[0.9, 0.1, 0.0, 0.0],
)
true_bbox_data_Z10 = BboxData(image_path=image_path, xmin=435, ymin=275, xmax=591, ymax=440, label="Z")
pred_bbox_data_Z10 = BboxData(
    image_path=image_path,
    xmin=455,
    ymin=312,
    xmax=567,
    ymax=420,
    label="Z",
    top_n=4,
    labels_top_n=["Z", "A", "B", "other"],
    classification_scores_top_n=[0.8, 0.1, 0.0, 0.1],
)

true_bbox_data_C11 = BboxData(image_path=image_path, xmin=435, ymin=525, xmax=591, ymax=690, label="C")
pred_bbox_data_B11 = BboxData(
    image_path=image_path,
    xmin=455,
    ymin=562,
    xmax=567,
    ymax=670,
    label="B",
    top_n=4,
    labels_top_n=["B", "A", "Z", "other"],
    classification_scores_top_n=[0.4, 0.4, 0.1, 0.1],
)


true_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        true_bbox_data_A1,
        true_bbox_data_A2,
        true_bbox_data_A3,
        true_bbox_data_B4,
        true_bbox_data_other6,
        true_bbox_data_B7,
        true_bbox_data_other8,
        true_bbox_data_Z10,
        true_bbox_data_C11,
    ],
)
pred_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        pred_bbox_data_A1,
        pred_bbox_data_B2,
        pred_bbox_data_Z3,
        pred_bbox_data_B4,
        pred_bbox_data_B6,
        pred_bbox_data_other7,
        pred_bbox_data_A8,
        pred_bbox_data_Z10,
        pred_bbox_data_B11,
    ],
)


def test_pipeline_metrics():
    image = visualize_images_data_side_by_side(
        image_data1=true_image_data, image_data2=pred_image_data, use_labels1=True, use_labels2=True, overlay=True
    )
    df_classification_metrics = get_df_classification_metrics(
        n_true_bboxes_data=[true_image_data.bboxes_data],
        n_pred_bboxes_data=[pred_image_data.bboxes_data],
        pseudo_class_names=["other"],
        known_class_names=["A", "B", "Z", "other"],
        tops_n=[1, 2, 3, 4],
    )
    df_classification_metrics = df_classification_metrics.loc[
        [
            "A",
            "B",
            "C",
            "Z",
            "other",
            "all_accuracy",
            "all_weighted_average",
            "all_accuracy_without_pseudo_classes",
            "all_weighted_average_without_pseudo_classes",
            "known_accuracy_without_pseudo_classes",
            "known_weighted_average_without_pseudo_classes",
        ]
    ]
    image_bytes = BytesIO()
    dfi.export(
        obj=df_classification_metrics,
        fontsize=10,
        filename=image_bytes,
        table_conversion="matplotlib",
    )
    df_image = imageio.v3.imread(image_bytes.getvalue())

    total_image = concat_images(image_a=image, image_b=df_image, how="vertically", mode="RGB")

    Image.fromarray(total_image).save(test_dir / "df_classification_metrics_A_B_Z_visualized.jpg")
