import subprocess


def get_gpu_memory():
    """
    获取所有 GPU 当前显存使用情况。
    """

    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
    ).decode("utf-8")

    gpu_info = []

    for line in result.strip().split("\n"):
        index, used, total = line.split(",")

        index = int(index.strip())
        used = int(used.strip())
        total = int(total.strip())

        free = total - used

        gpu_info.append({
            "index": index,
            "used": used,
            "total": total,
            "free": free
        })

    return gpu_info


def get_free_gpu():
    """
    自动选择当前剩余显存最多的 GPU。
    """

    gpu_info = get_gpu_memory()

    gpu_info.sort(
        key=lambda x: x["free"],
        reverse=True
    )

    best_gpu = gpu_info[0]

    return best_gpu["index"]


if __name__ == "__main__":

    print("========== GPU Information ==========")

    gpus = get_gpu_memory()

    for gpu in gpus:
        print(
            f"GPU {gpu['index']}: "
            f"used={gpu['used']} MiB, "
            f"total={gpu['total']} MiB, "
            f"free={gpu['free']} MiB"
        )

    selected_gpu = get_free_gpu()

    print()
    print(f"Selected GPU: {selected_gpu}")

