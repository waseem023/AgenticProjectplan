import PyPDF2
from sentence_transformers import SentenceTransformer
from experiments.experiment_runner import ExperimentRunner

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def chunk_text(text, max_chunk_size=512):
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def main():
    requirements_text = read_pdf("data/Requirements_Specification.pdf")
    tasks_text = read_pdf("data/SampleProjectTasksEstimates.pdf")

    requirements_chunks = chunk_text(requirements_text)
    tasks_chunks = chunk_text(tasks_text)

    requirements_text = " ".join(requirements_chunks)
    tasks_text = " ".join(tasks_chunks)

    experiment_runner = ExperimentRunner()

    for i in range(1, 9):
        print(f"\nRunning Experiment {i}...")
        requirements, tasks_estimates = experiment_runner.run_experiment(i, requirements_text, tasks_text)

        print(f"Experiment {i} Generated Requirements:")
        for req in requirements:
            print(req)
        
        print(f"\nExperiment {i} Generated Tasks and Estimates:")
        for role, estimates in tasks_estimates.items():
            print(f"\n{role}:")
            for estimate in estimates:
                print(estimate)

if __name__ == "__main__":
    main()