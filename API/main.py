from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(title="AI_Tongue", description="API for AI_Tongue", version="0.1")
# 让app可以跨域
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "/home/sc/LLM_Learning/checkpoints/Qwen2-VL-7B-Instruct-sft-lora-Tongue"

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)


class OpenAIRequest(BaseModel):
    model: str
    messages: list
    temperature: Optional[float] = 0.01
    max_tokens: Optional[int] = 8192


@app.post("/v1/chat/completions")
async def generate_completion(request: OpenAIRequest):
    try:
        # Validate messages format
        for message in request.messages:
            if not isinstance(message, dict) or "content" not in message:
                raise HTTPException(status_code=400, detail="Invalid message format")

        # Prepare the input messages for processing
        text = processor.apply_chat_template(
            [
                {"role": msg["role"], "content": msg["content"]}
                for msg in request.messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(request.messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output
        generated_ids = model.generate(
            **inputs, max_new_tokens=request.max_tokens, temperature=request.temperature
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Format the response
        response = {
            "id": "cmpl-unique-id",
            "object": "text_completion",
            "created": datetime.now().timestamp(),
            "model": request.model,
            "choices": [
                {
                    "text": output_text[0],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
