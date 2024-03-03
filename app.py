from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.intent_handler import IntentHandler

app = FastAPI()


class Query(BaseModel):
    """Query model to handle the user query"""

    user_query: str = Field(description="User query to process", default=None)


@app.post("/intent_query/")
async def process_query(query: Query):
    """Endpoint to handle the intent execution and return the response to the
    user.\n
    All the responses will be structured same as defined by the corresponding
    agent in charge or the intent execution by using the `IntentHandler` class.\n
    Finally the response will be returned with the format defined by the
    `ResponseTemplate` object.\n
    **Note**: The response will be in JSON format which will be used by the streamlit
    app to display the response.
    """
    try:
        intent_handler = IntentHandler()
        intent_handler.user_input = query.user_query
        raw_response = intent_handler.run()
        final_response = raw_response.template_structure
        return {"response": final_response}
    except Exception as e:
        return {"error": str(e)}
