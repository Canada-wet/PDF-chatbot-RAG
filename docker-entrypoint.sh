#!/bin/bash

# can export secrets here, usually the secrets will be from an access control tool such as vault
# set path to a vault script and export secrets here in production

# print out important information
echo "INFO: Starting the apploications..."

# Execute Streamlit app
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
