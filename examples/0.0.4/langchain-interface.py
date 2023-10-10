from dotenv import load_dotenv
import os

from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

load_dotenv()

api_endpoint = 'https://us-south.ml.cloud.ibm.com'
api_key = os.getenv('API_KEY')
project_id = os.getenv('PROJECT_ID')

creds = Credentials(
    api_key=api_key,
    api_endpoint=api_endpoint,
    project_id=project_id
)

parameters = {
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0,
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE
}

llm = LangChainInterface(
    model=ModelTypes.FLAN_UL2.value,
    params=parameters,
    credentials=creds
)

prompt = "What is life?"
output = llm(prompt)
print(prompt)
print(output)

#### Example output
#
# What is life?
# Perception of self