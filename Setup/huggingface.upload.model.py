import os
from huggingface_hub import create_repo, upload_folder

def main():
    # Replace with your desired repository id (e.g., "your-username/your-model-repo")
    repo_id: str = "jedlee2004/physics-chat"
    
    # Path to the folder containing your fine-tuned model and tokenizer (output folder)
    output_folder: str = "./output"
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"Repository '{repo_id}' created (or already exists).")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload the folder to the repository on Hugging Face Hub
    try:
        print(f"Uploading contents of '{output_folder}' to repo '{repo_id}' ...")
        upload_folder(
            repo_id=repo_id,
            folder_path=output_folder,
            commit_message="Upload fine-tuned model and tokenizer"
        )
        print("Upload complete!")
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    main()
