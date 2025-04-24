import json

# Paths
INPUT_FILE = "/data/scripts/fedlex_documents.txt"
OUTPUT_FILE = "legal_instructions.jsonl"

# Load and convert
def convert_fedlex_to_instruction_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    samples = []
    for doc in content.split("---\n"):
        if doc.strip():
            parts = doc.split("\n", 2)
            if len(parts) >= 3:
                doc_id = parts[0].strip()
                title = parts[1].strip()
                body = parts[2].strip()

                # Example Q&A pairs for instruction tuning
                samples.append({
                    "instruction": f"What is the legal subject of the document titled '{title}'?",
                    "context": body[:1000],
                    "response": f"The legal subject of '{title}' is related to Swiss federal law under ID {doc_id}."
                })

                samples.append({
                    "instruction": f"Summarize the key points of the document '{title}'.",
                    "context": body[:1200],
                    "response": f"The document '{title}' outlines the following key legal points: ..."
                })

                samples.append({
                    "instruction": f"Which jurisdiction does the document '{title}' apply to?",
                    "context": body[:1000],
                    "response": f"The document '{title}' applies to federal jurisdiction unless otherwise specified."
                })

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"? Converted {len(samples)} samples and saved to {output_path}")

# Run conversion
convert_fedlex_to_instruction_format(INPUT_FILE, OUTPUT_FILE)
