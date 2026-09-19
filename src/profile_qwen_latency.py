import time
import json
import psutil
import threading
from pathlib import Path
from llama_cpp import Llama


MODELS = {
    "0.5B": "models/qwen-2.5-0.5b-instruct.gguf",
    "1.5B": "models/qwen-2.5-1.5b-instruct.gguf",
}

PROMPT = """You are an aquaculture assistant.
Answer this question briefly and clearly:

What are the common causes of low dissolved oxygen in aquaculture ponds?
"""


process = psutil.Process()


def get_memory_mb():
    return process.memory_info().rss / (1024 * 1024)


def monitor_resources(stop_event, measurements):
    while not stop_event.is_set():
        measurements.append({
            "memory_mb": get_memory_mb(),
            "cpu_percent": process.cpu_percent(interval=0.1)
        })


results = {}

for model_name, model_path in MODELS.items():
    print(f"\nLoading {model_name} model...")

    memory_before = get_memory_mb()
    load_start = time.perf_counter()

    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        verbose=False
    )

    load_time = time.perf_counter() - load_start
    memory_after_load = get_memory_mb()

    print(f"{model_name} loaded in {load_time:.2f} seconds")
    print("Generating response...")

    measurements = []
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(stop_event, measurements)
    )

    process.cpu_percent(interval=None)
    monitor_thread.start()

    generation_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    response_text = ""

    stream = llm(
        PROMPT,
        max_tokens=100,
        stream=True
    )

    for output in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()

        text = output["choices"][0]["text"]
        response_text += text
        token_count += 1

    generation_end = time.perf_counter()

    stop_event.set()
    monitor_thread.join()

    ttft = first_token_time - generation_start
    total_generation_time = generation_end - generation_start

    peak_memory = max(
        item["memory_mb"] for item in measurements
    )

    average_cpu = sum(
        item["cpu_percent"] for item in measurements
    ) / len(measurements)

    peak_cpu = max(
        item["cpu_percent"] for item in measurements
    )

    results[model_name] = {
        "model_path": model_path,
        "model_load_time_seconds": round(load_time, 4),
        "memory_before_load_mb": round(memory_before, 2),
        "memory_after_load_mb": round(memory_after_load, 2),
        "memory_increase_mb": round(
            memory_after_load - memory_before, 2
        ),
        "peak_memory_during_generation_mb": round(
            peak_memory, 2
        ),
        "average_cpu_percent": round(average_cpu, 2),
        "peak_cpu_percent": round(peak_cpu, 2),
        "time_to_first_token_seconds": round(ttft, 4),
        "total_generation_time_seconds": round(
            total_generation_time, 4
        ),
        "stream_chunks": token_count,
        "response": response_text.strip()
    }

    print(f"TTFT: {ttft:.2f} seconds")
    print(f"Total generation time: {total_generation_time:.2f} seconds")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Average CPU: {average_cpu:.2f}%")
    print(f"Peak CPU: {peak_cpu:.2f}%")

    del llm


output_path = Path("data/evaluation/qwen_latency_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(f"\nResults saved to: {output_path}")