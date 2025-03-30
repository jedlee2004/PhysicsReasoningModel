import os
from datasets import load_dataset

#https://huggingface.co/datasets/Josephgflowers/Par-Four-Fineweb-Edu-Fortified-Chemistry-Physics-Astronomy-Math-Reason

# Ensure the datasets folder exists
os.makedirs("datasets", exist_ok=True)

# https://huggingface.co/datasets/camel-ai/physics
camel_physics = load_dataset("camel-ai/physics")
camel_physics.save_to_disk("datasets/camel_physics")

# https://huggingface.co/datasets/gallen881/arxiv-physics
arxiv_physics = load_dataset("gallen881/arxiv-physics")
arxiv_physics.save_to_disk("datasets/arxiv_physics")

#https://huggingface.co/datasets/Akul/alpaca_physics_dataset
alpaca_physics = load_dataset("Akul/alpaca_physics_dataset")
alpaca_physics.save_to_disk("datasets/alpaca_physics")

