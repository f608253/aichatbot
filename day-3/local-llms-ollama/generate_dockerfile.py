import ollama

PROMPT = """
ONLY Generate an ideal Dockerfile for {language} with best practices. Don't provide any description
Please note to use distroless image.
Include:
- Base image
- Installing dependencies
- Setting working directory
- Adding source code
- Giving 755 permission to the app.jar file in container
- Running the application
- Multistage docker build
"""

def generate_dockerfile(language):
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': PROMPT.format(language=language)}])
    return response['message']['content']

if __name__ == '__main__':
    language = input("Enter the programming language: ")
    dockerfile = generate_dockerfile(language)
    print("\nGenerated Dockerfile:\n")
    print(dockerfile)