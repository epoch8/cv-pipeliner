import dataframe_image as dfi
from pathlib import Path
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.visualizers.core.image_data import visualize_images_data_side_by_side
from cv_pipeliner.visualizers.core.image_data_matching import visualize_image_data_matching_side_by_side

from cv_pipeliner.metrics.pipeline import get_df_pipeline_metrics

test_dir = Path(__file__).parent / 'images'
image_path = Path(__file__).parent / 'original.jpg'

# Banana
true_bbox_data_banana = BboxData(  # detection: FN, pipeline: FN for banana
    image_path=image_path,
    xmin=296,
    ymin=104,
    xmax=1190,
    ymax=650,
    label='banana'
)
pred_bbox_data_banana_part1 = BboxData(  # detection: FP, pipeline: FP for banana
    image_path=image_path,
    xmin=314,
    ymin=122,
    xmax=501,
    ymax=314,
    label='banana'
)
pred_bbox_data_banana_part2 = BboxData(  # detection: FP
                                         # pipeline: TP when extra_bbox_label == 'trash'
                                         # pipeline: FP when extra_bbox_label is not 'trash'

    image_path=image_path,
    xmin=536,
    ymin=368,
    xmax=1172,
    ymax=542,
    label='other'
)

# Top apple
true_bbox_data_top_apple = BboxData(  # detection: FN
    image_path=image_path,
    xmin=1254,
    ymin=337,
    xmax=1515,
    ymax=606,
    label='apple'
)

# Bottom apple
true_bbox_data_bottom_apple = BboxData(  # <-> pred_bbox_data_bottom_apple; detection: FN, pipeline: FN for apple
    image_path=image_path,
    xmin=445,
    ymin=701,
    xmax=742,
    ymax=1021,
    label='apple'
)
pred_bbox_data_bottom_apple = BboxData(  # <-> true_bbox_data_bottom_apple; detection: TP, pipeline: TP for apple
    image_path=image_path,
    xmin=463,
    ymin=709,
    xmax=736,
    ymax=1011,
    label='apple'
)

# Mango
true_bbox_data_mango = BboxData(  # <-> pred_bbox_data_mango_insid; detection: TP, pipeline: FP
    image_path=image_path,
    xmin=924,
    ymin=622,
    xmax=1436,
    ymax=1034,
    label='mango'
)
pred_bbox_data_mango_outside = BboxData(  # detection: TP, pipeline: FP for banana
    image_path=image_path,
    xmin=933,
    ymin=640,
    xmax=1419,
    ymax=1011,
    label='banana'
)
pred_bbox_data_mango_inside = BboxData(  # <-> true_bbox_data_mango; detection: FP, pipeline: FP for mango
    image_path=image_path,
    xmin=998,
    ymin=660,
    xmax=1383,
    ymax=1013,
    label='mango'
)

# Extra bboxes
pred_bbox_data_extra_bbox1 = BboxData(  # detection: FP
                                        # pipeline: TP when extra_bbox_label == 'trash'
                                        # pipeline: FP when extra_bbox_label is not 'trash'
    image_path=image_path,
    xmin=1599,
    ymin=486,
    xmax=1822,
    ymax=698,
    label='other'
)
pred_bbox_data_extra_bbox2 = BboxData(  # detection: FP, pipeline: FP for apple
    image_path=image_path,
    xmin=1602,
    ymin=788,
    xmax=1835,
    ymax=1013,
    label='apple'
)


true_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        true_bbox_data_banana,
        true_bbox_data_mango,
        true_bbox_data_top_apple,
        true_bbox_data_bottom_apple,
    ]
)
pred_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        pred_bbox_data_banana_part1,
        pred_bbox_data_banana_part2,
        pred_bbox_data_bottom_apple,
        pred_bbox_data_mango_outside,
        pred_bbox_data_mango_inside,
        pred_bbox_data_extra_bbox1,
        pred_bbox_data_extra_bbox2
    ]
)

Image.fromarray(
    visualize_images_data_side_by_side(
        image_data1=true_image_data,
        image_data2=pred_image_data,
        use_labels1=True,
        use_labels2=True
    )
).save(test_dir / 'original_visualized.jpg')


def test_image_data_matching_detection():
    minimum_iou = 0.5

    image_data_matching = ImageDataMatching(
        true_image_data=true_image_data,
        pred_image_data=pred_image_data,
        minimum_iou=minimum_iou
    )

    Image.fromarray(
        visualize_image_data_matching_side_by_side(
            image_data_matching=image_data_matching,
            error_type='detection',
            true_use_labels=True,
            pred_use_labels=True
        )
    ).save(test_dir / 'detection.jpg')

    # Banana tests
    bbox_data_matching_banana = image_data_matching.find_bbox_data_matching(true_bbox_data_banana, tag='true')
    bbox_data_matching_banana_part1 = image_data_matching.find_bbox_data_matching(pred_bbox_data_banana_part1,
                                                                                  tag='pred')
    bbox_data_matching_banana_part2 = image_data_matching.find_bbox_data_matching(pred_bbox_data_banana_part2,
                                                                                  tag='pred')

    assert bbox_data_matching_banana.pred_bbox_data is None
    assert bbox_data_matching_banana.get_detection_error_type() == "FN"

    assert bbox_data_matching_banana_part1.true_bbox_data is None
    assert bbox_data_matching_banana_part1.get_detection_error_type() == "FP"

    assert bbox_data_matching_banana_part2.true_bbox_data is None
    assert bbox_data_matching_banana_part2.get_detection_error_type() == "FP"

    # Top apple tests
    bbox_data_matching_top_apple = image_data_matching.find_bbox_data_matching(true_bbox_data_top_apple,
                                                                               tag='true')

    assert bbox_data_matching_top_apple.pred_bbox_data is None
    assert bbox_data_matching_top_apple.get_detection_error_type() == "FN"

    # Bottom apple tests
    bbox_data_matching_bottom_apple_true = image_data_matching.find_bbox_data_matching(true_bbox_data_bottom_apple,
                                                                                       tag='true')
    bbox_data_matching_bottom_apple_pred = image_data_matching.find_bbox_data_matching(pred_bbox_data_bottom_apple,
                                                                                       tag='pred')

    assert bbox_data_matching_bottom_apple_true == bbox_data_matching_bottom_apple_pred
    assert bbox_data_matching_bottom_apple_true.get_detection_error_type() == "TP"

    # Mango tests
    bbox_data_matching_mango = image_data_matching.find_bbox_data_matching(true_bbox_data_mango, tag='true')
    bbox_data_matching_mango_outside = image_data_matching.find_bbox_data_matching(pred_bbox_data_mango_outside,
                                                                                   tag='pred')
    bbox_data_matching_mango_inside = image_data_matching.find_bbox_data_matching(pred_bbox_data_mango_inside,
                                                                                  tag='pred')

    assert bbox_data_matching_mango == bbox_data_matching_mango_outside
    assert bbox_data_matching_mango.get_detection_error_type() == "TP"

    assert bbox_data_matching_mango_inside.true_bbox_data is None
    assert bbox_data_matching_mango_inside.get_detection_error_type() == "FP"

    # Extra bboxes tests
    bbox_data_matching_extra_bbox1 = image_data_matching.find_bbox_data_matching(pred_bbox_data_extra_bbox1,
                                                                                 tag='pred')
    bbox_data_matching_extra_bbox2 = image_data_matching.find_bbox_data_matching(pred_bbox_data_extra_bbox2,
                                                                                 tag='pred')

    assert bbox_data_matching_extra_bbox1.true_bbox_data is None
    assert bbox_data_matching_extra_bbox1.get_detection_error_type() == "FP"

    assert bbox_data_matching_extra_bbox2.true_bbox_data is None
    assert bbox_data_matching_extra_bbox2.get_detection_error_type() == "FP"


def test_image_data_matching_pipeline():

    minimum_iou = 0.5
    tag_default = 'all'
    tag_with_extra_bbox_label_other = 'all_with_extra_bbox_label_other'
    tag_known_banana_apple = 'known_banana_apple'
    tag_known_banana_apple_with_extra_bbox_label_other = 'known_banana_apple_with_extra_bbox_label_other'

    for tag in [
        tag_default, tag_with_extra_bbox_label_other, tag_known_banana_apple,
        tag_known_banana_apple_with_extra_bbox_label_other
    ]:
        extra_bbox_label = 'trash'
        known_class_names = None
        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            extra_bbox_label = 'other'
        if tag in [tag_known_banana_apple, tag_known_banana_apple_with_extra_bbox_label_other]:
            known_class_names = ['banana', 'apple']

        image_data_matching = ImageDataMatching(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label,
        )

        # Banana tests
        bbox_data_matching_banana = image_data_matching.find_bbox_data_matching(true_bbox_data_banana, tag='true')
        bbox_data_matching_banana_part1 = image_data_matching.find_bbox_data_matching(pred_bbox_data_banana_part1,
                                                                                      tag='pred')
        bbox_data_matching_banana_part2 = image_data_matching.find_bbox_data_matching(pred_bbox_data_banana_part2,
                                                                                      tag='pred')

        assert bbox_data_matching_banana.pred_bbox_data is None
        assert bbox_data_matching_banana.get_pipeline_error_type() == "FN"
        assert bbox_data_matching_banana.get_pipeline_error_type(label='apple') == "TN"
        assert bbox_data_matching_banana.get_pipeline_error_type(label='banana') == "FN"
        assert bbox_data_matching_banana.get_pipeline_error_type(label='mango') == "TN"

        assert bbox_data_matching_banana_part1.true_bbox_data is None
        assert bbox_data_matching_banana_part1.get_pipeline_error_type() == "FP (extra bbox)"
        assert bbox_data_matching_banana_part1.get_pipeline_error_type(label='apple') == "TN (extra bbox)"
        assert bbox_data_matching_banana_part1.get_pipeline_error_type(label='banana') == "FP (extra bbox)"
        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            assert bbox_data_matching_banana_part1.get_pipeline_error_type(label='other') == "FN (extra bbox)"
        else:
            assert bbox_data_matching_banana_part1.get_pipeline_error_type(label='other') == "TN (extra bbox)"

        assert bbox_data_matching_banana_part2.true_bbox_data is None
        assert bbox_data_matching_banana_part2.get_pipeline_error_type(label='apple') == "TN (extra bbox)"
        assert bbox_data_matching_banana_part2.get_pipeline_error_type(label='banana') == "TN (extra bbox)"
        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            assert bbox_data_matching_banana_part2.get_pipeline_error_type() == "TP (extra bbox)"
            assert bbox_data_matching_banana_part2.get_pipeline_error_type(label='other') == "TP (extra bbox)"
        else:
            assert bbox_data_matching_banana_part2.get_pipeline_error_type() == "FP (extra bbox)"
            assert bbox_data_matching_banana_part2.get_pipeline_error_type(label='other') == "FP (extra bbox)"

        # Top apple tests
        bbox_data_matching_top_apple = image_data_matching.find_bbox_data_matching(true_bbox_data_top_apple,
                                                                                   tag='true')

        assert bbox_data_matching_top_apple.pred_bbox_data is None
        assert bbox_data_matching_top_apple.get_pipeline_error_type() == "FN"

        # Bottom apple tests
        bbox_data_matching_bottom_apple_true = image_data_matching.find_bbox_data_matching(true_bbox_data_bottom_apple,
                                                                                           tag='true')
        bbox_data_matching_bottom_apple_pred = image_data_matching.find_bbox_data_matching(pred_bbox_data_bottom_apple,
                                                                                           tag='pred')

        assert bbox_data_matching_bottom_apple_true == bbox_data_matching_bottom_apple_pred
        assert bbox_data_matching_bottom_apple_true.get_pipeline_error_type() == "TP"
        assert bbox_data_matching_bottom_apple_true.get_pipeline_error_type(label='apple') == "TP"
        assert bbox_data_matching_bottom_apple_true.get_pipeline_error_type(label='banana') == "TN"
        assert bbox_data_matching_bottom_apple_true.get_pipeline_error_type(label='mango') == "TN"
        assert bbox_data_matching_bottom_apple_true.get_pipeline_error_type(label='other') == "TN"

        # Mango tests
        bbox_data_matching_mango = image_data_matching.find_bbox_data_matching(true_bbox_data_mango, tag='true')
        bbox_data_matching_mango_outside = image_data_matching.find_bbox_data_matching(pred_bbox_data_mango_outside,
                                                                                       tag='pred')
        bbox_data_matching_mango_inside = image_data_matching.find_bbox_data_matching(pred_bbox_data_mango_inside,
                                                                                      tag='pred')

        assert bbox_data_matching_mango == bbox_data_matching_mango_outside
        assert bbox_data_matching_mango.get_pipeline_error_type(label='apple') == "TN"
        assert bbox_data_matching_mango.get_pipeline_error_type(label='mango') == "FN"
        assert bbox_data_matching_mango.get_pipeline_error_type(label='banana') == "FP"
        assert bbox_data_matching_mango.get_pipeline_error_type(label='other') == "TN"

        assert bbox_data_matching_mango_inside.true_bbox_data is None
        assert bbox_data_matching_mango_inside.get_pipeline_error_type() == "FP (extra bbox)"
        assert bbox_data_matching_mango_inside.get_pipeline_error_type(label='apple') == "TN (extra bbox)"
        assert bbox_data_matching_mango_inside.get_pipeline_error_type(label='banana') == "TN (extra bbox)"
        assert bbox_data_matching_mango_inside.get_pipeline_error_type(label='mango') == "FP (extra bbox)"
        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            assert bbox_data_matching_mango_inside.get_pipeline_error_type(label='other') == "FN (extra bbox)"
        else:
            assert bbox_data_matching_mango_inside.get_pipeline_error_type(label='other') == "TN (extra bbox)"

        # Extra bboxes tests
        bbox_data_matching_extra_bbox1 = image_data_matching.find_bbox_data_matching(pred_bbox_data_extra_bbox1,
                                                                                     tag='pred')
        bbox_data_matching_extra_bbox2 = image_data_matching.find_bbox_data_matching(pred_bbox_data_extra_bbox2,
                                                                                     tag='pred')

        assert bbox_data_matching_extra_bbox1.true_bbox_data is None
        assert bbox_data_matching_extra_bbox1.get_pipeline_error_type(label='apple') == "TN (extra bbox)"
        assert bbox_data_matching_extra_bbox1.get_pipeline_error_type(label='banana') == "TN (extra bbox)"
        assert bbox_data_matching_extra_bbox1.get_pipeline_error_type(label='mango') == "TN (extra bbox)"

        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            assert bbox_data_matching_extra_bbox1.get_pipeline_error_type(label='other') == "TP (extra bbox)"
        else:
            assert bbox_data_matching_extra_bbox1.get_pipeline_error_type(label='other') == "FP (extra bbox)"

        assert bbox_data_matching_extra_bbox2.true_bbox_data is None
        assert bbox_data_matching_extra_bbox2.get_pipeline_error_type(label='apple') == "FP (extra bbox)"
        assert bbox_data_matching_extra_bbox2.get_pipeline_error_type(label='banana') == "TN (extra bbox)"
        assert bbox_data_matching_extra_bbox2.get_pipeline_error_type(label='mango') == "TN (extra bbox)"

        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            assert bbox_data_matching_extra_bbox2.get_pipeline_error_type(label='other') == "FN (extra bbox)"
        else:
            assert bbox_data_matching_extra_bbox2.get_pipeline_error_type(label='other') == "TN (extra bbox)"

        df_pipeline_metrics = get_df_pipeline_metrics(
            true_images_data=[true_image_data],
            pred_images_data=[pred_image_data],
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label,
            pseudo_class_names=['trash', 'other'],
            known_class_names=known_class_names
        )

        assert df_pipeline_metrics.loc['apple', 'TP'] == 1
        assert df_pipeline_metrics.loc['apple', 'FP'] == 0
        assert df_pipeline_metrics.loc['apple', 'FN'] == 1
        assert df_pipeline_metrics.loc['apple', 'TP (extra bbox)'] == 0
        assert df_pipeline_metrics.loc['apple', 'FP (extra bbox)'] == 1
        assert df_pipeline_metrics.loc['apple', 'FN (extra bbox)'] == 0

        assert df_pipeline_metrics.loc['banana', 'TP'] == 0
        assert df_pipeline_metrics.loc['banana', 'FP'] == 1
        assert df_pipeline_metrics.loc['banana', 'FN'] == 1
        assert df_pipeline_metrics.loc['banana', 'TP (extra bbox)'] == 0
        assert df_pipeline_metrics.loc['banana', 'FP (extra bbox)'] == 1
        assert df_pipeline_metrics.loc['banana', 'FN (extra bbox)'] == 0

        assert df_pipeline_metrics.loc['mango', 'TP'] == 0
        assert df_pipeline_metrics.loc['mango', 'FP'] == 0
        assert df_pipeline_metrics.loc['mango', 'FN'] == 1
        assert df_pipeline_metrics.loc['mango', 'TP (extra bbox)'] == 0
        assert df_pipeline_metrics.loc['mango', 'FP (extra bbox)'] == 1
        assert df_pipeline_metrics.loc['mango', 'FN (extra bbox)'] == 0

        if tag in [tag_with_extra_bbox_label_other, tag_known_banana_apple_with_extra_bbox_label_other]:
            df_pipeline_metrics.loc['other', 'TP'] == 0
            df_pipeline_metrics.loc['other', 'FP'] == 0
            df_pipeline_metrics.loc['other', 'FN'] == 0
            df_pipeline_metrics.loc['other', 'TP (extra bbox)'] == 0
            df_pipeline_metrics.loc['other', 'FP (extra bbox)'] == 0
            df_pipeline_metrics.loc['other', 'FN (extra bbox)'] == 0
            df_pipeline_metrics.loc['other (extra bbox)', 'TP'] == 0
            df_pipeline_metrics.loc['other (extra bbox)', 'FP'] == 0
            df_pipeline_metrics.loc['other (extra bbox)', 'FN'] == 0
            df_pipeline_metrics.loc['other (extra bbox)', 'TP (extra bbox)'] == 2
            df_pipeline_metrics.loc['other (extra bbox)', 'FP (extra bbox)'] == 0
            df_pipeline_metrics.loc['other (extra bbox)', 'FN (extra bbox)'] == 3

        for label in [None, 'apple', 'banana', 'mango', 'other']:
            Image.fromarray(
                visualize_image_data_matching_side_by_side(
                    image_data_matching=image_data_matching,
                    error_type='pipeline',
                    true_use_labels=True,
                    pred_use_labels=True,
                    label=label
                )
            ).save(test_dir / f'pipeline_tag={tag}_label={label}.jpg')
        dfi.export(
            obj=df_pipeline_metrics,
            filename=str(test_dir / f"df_metrics_pipeline_tag={tag}.png"),
            table_conversion='matplotlib'
        )
