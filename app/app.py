import streamlit as st
import numpy as np
import imageio
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.visualizers.core.images_data import visualize_image_data

from model import name_to_detection_model_spec, name_to_classification_model_spec, load_pipeline_inferencer

st.set_option('deprecation.showfileUploaderEncoding', False)

data_load_state = st.text('Choose model:')
detection_model_name = st.selectbox(
    label='Detection',
    options=[None] + [name for name in name_to_detection_model_spec]
)
classification_model_name = st.selectbox(
    label='Classification',
    options=[None] + [name for name in name_to_classification_model_spec]
)
if detection_model_name is not None and classification_model_name is not None: 
    data_load_state.text("Loading models...")
    pipeline_inferencer = load_pipeline_inferencer(detection_model_name, classification_model_name)
    data_load_state.text("Done!")

    st.title(f'Pipeline')

    @st.cache(allow_output_mutation=True)
    def inference_one_image(uploaded_file):
        image = np.array(imageio.imread(uploaded_file))
        image_data = ImageData(image=image)
        image_data_gen = BatchGeneratorImageData([image_data], batch_size=1)
        pred_image_data = pipeline_inferencer.predict(image_data_gen, detection_score_threshold=0.4)[0]
        pred_image_data.image = image_data.image
        return pred_image_data

    uploaded_file = st.file_uploader("Drop an image", type=["png", "jpeg", "jpg"])

    if uploaded_file is not None:
        pred_image_data = inference_one_image(uploaded_file)
        pred_image = visualize_image_data(pred_image_data)
        st.image(image=[uploaded_file, pred_image], use_column_width=True)

        st.image(image=[bbox_data.image_bbox for bbox_data in pred_image_data.bboxes_data],
                 caption=[bbox_data.label for bbox_data in pred_image_data.bboxes_data])
        for bbox_data in pred_image_data.bboxes_data:
            st.image(image=bbox_data.image_bbox)
            st.text(bbox_data.label)
            '----'
