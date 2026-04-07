import glob
from pathlib import Path


def fetch_documents(knowledge_base_path: Path):
    folders = glob.glob(f"{knowledge_base_path}/*")
    for folder in folders:
        doc_type = Path(folder).name
        print(f"Document type: {doc_type}")

    # print(f"Found {len(files)} markdown files in the knowledge base.")

    # all_documents = ""

    # for file in files:
    #     with open(file, "r", encoding="utf-8") as f:
    #         content = f.read()
    #         all_documents += content + "\n\n"

    # return all_documents

    return "test"