import os
import streamlit as st
from two_stage_pipeliner.app.app import run_app

if 'TWO_STAGE_PIPELINER_APP_CONFIG' in os.environ:
    run_app(os.environ['TWO_STAGE_PIPELINER_APP_CONFIG'])
else:
    st.warning(
        "Environment variable 'TWO_STAGE_PIPELINER_APP_CONFIG' was not found. Loading default config instead."
    )
    run_app('app_config.yaml')
