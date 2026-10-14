import json
import os

from groq import Groq

from src.schemas import ExtractionResult


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return []

    if not GROQ_API_KEY:
        raise RuntimeError(
            "Set the GROQ_API_KEY environment variable before querying."
        )

    context = "\n\n".join(
        (
            f"Source page {chunk['metadata']['page']}:\n"
            f"{chunk['text']}"
        )
        for chunk in retrieved_chunks
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract vehicle specifications from service manual "
                    "context. Use only supplied context. Return only valid "
                    "JSON matching this schema: "
                    '{"specifications":[{"specification_name":"string",'
                    '"value":"string","unit":"string or null",'
                    '"vehicle_model":"string or null",'
                    '"part_number":"string or null","source_page":1,'
                    '"source_text":"string","confidence":0.0}]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Manual context:\n{context}"
                ),
            },
        ],
    )

    response_text = response.choices[0].message.content
    if not response_text:
        raise RuntimeError("Groq returned an empty answer.")

    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.removeprefix("```json").strip()
        response_text = response_text.removesuffix("```").strip()

    try:
        result = ExtractionResult.model_validate(json.loads(response_text))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "Groq returned invalid structured specification JSON."
        ) from exc

    return [item.model_dump() for item in result.specifications]
