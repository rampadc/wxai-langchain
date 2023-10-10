from dotenv import load_dotenv
import os

from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

from langchain import PromptTemplate

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

# An example prompt with no input variables
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
no_input_prompt.format()
# -> "Tell me a joke."

# An example prompt with one input variable
one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
one_input_prompt.format(adjective="funny")
# -> "Tell me a funny joke."

# An example prompt with multiple input variables
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"], 
    template="Tell me a {adjective} joke about {content}."
)

prompt = multiple_input_prompt.format(adjective="funny", content="chickens")
output = llm(prompt)

print(prompt)
print(output)

### Example output:
#
# Tell me a funny joke about chickens.
# The chicken walked into the kitchen and sat down.