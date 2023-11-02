# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import numpy as np
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

class Model:
    def __init__(self):
        pass
    
    def predict(self, instances):
        predictions = []
        for i in instances:
            predictions.append(2)

        return np.array(predictions)

def classify(instances):
    model=Model()
    classes = model.predict(instances)
    return classes

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
        #d = {'KNN': ["Nearest Neighbor", "Supervised"], 'Decision Tree': ["Decision Tree", "Supervision"], 'SVM': ["Support Vector Machine", "Supervision"]},
    )

    st.write("# :white_flower: Welcome to the Iris Classifier 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        This app classifies Iris flowers into four classes.

        Input the values for each feature to classify your flower.
    """
    )

    sepal_length = st.text_input(label='sepal_length')
    sepal_width = st.text_input(label='sepal_width')
    petal_length = st.text_input(label='petal_length')
    petal_width = st.text_input(label='petal_width')

    if st.button('Submit'):
        st.write(f'The values you submitted are: ', sepal_length, sepal_width, petal_length, petal_width)
        user_iris = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        results = classify(user_iris)
        iris_classes = ['Iris-Setosa', 'Iris-versicolor', 'Iris-virginica']
        for i in results:
            st.write(f'Your iris is of type:  {iris_classes[i]}')

   

if __name__ == "__main__":
    run()
