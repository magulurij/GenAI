import spacy
import requests
from bs4 import BeautifulSoup
import openai
import json
import docx  # Handles Word documents

# Initialize NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

# Function to scrape job descriptions from a URL
def scrape_job_description(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract job descriptions based on the HTML <a> attributes
        job_links = soup.find_all('a', {'class': 'card-title-link'})
        descriptions = [link.get_text(strip=True) for link in job_links]
        return descriptions
    except Exception as e:
        raise ValueError(f"Error scraping job description: {e}")

# Function to extract text from a Word document (resume)
def extract_resume_text(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        raise ValueError(f"Error reading resume: {e}")

# Function to extract skills and responsibilities using NLP
def extract_skills_responsibilities(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["NOUN", "PROPN"]]
    return list(set(skills))  # Deduplicate the list

# Function to generate quiz questions using OpenAI's GPT
def generate_quiz(role, skills, levels=["Easy", "Intermediate", "Advanced"]):
    quiz = []
    for level in levels:
        prompt = (f"Create 3 {level} quiz questions for a {role} role based on these skills: "
                  f"{', '.join(skills)}. Provide options and mark the correct answer.")
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500
            )
            questions = response.choices[0].text.strip()
            quiz.append({
                "difficulty": level,
                "questions": questions
            })
        except Exception as e:
            print(f"Error generating quiz questions: {e}")
            continue
    return quiz

# Function to structure the quiz into JSON format
def structure_quiz(role, quiz_data):
    structured_quiz = {
        "role": role,
        "quiz": quiz_data
    }
    return json.dumps(structured_quiz, indent=4)

# Input parameters
resume_path = "C:\\Users\\srava\\Downloads\\Jyothsna M Data Engineer.docx"  # Path to your resume file
job_url = "https://example.com/job-role"  # Replace with the actual job URL containing the HTML <a> tags

try:
    # Step 1: Scrape job descriptions from URL
    print("Scraping job descriptions...")
    job_descriptions = scrape_job_description(job_url)
    print(f"Job Descriptions Found: {job_descriptions}")

    # Step 2: Extract resume text
    print("Extracting resume text...")
    resume_text = extract_resume_text(resume_path)

    # Step 3: Preprocess data
    print("Extracting skills from job descriptions and resume...")
    jd_skills = extract_skills_responsibilities(" ".join(job_descriptions))
    resume_skills = extract_skills_responsibilities(resume_text)
    all_skills = list(set(jd_skills + resume_skills))

    # Step 4: Generate quiz questions
    role = "Senior Data Engineer"  # Replace with the desired role
    print("Generating quiz questions...")
    quiz_data = generate_quiz(role, all_skills)

    # Step 5: Structure and save the quiz
    print("Structuring quiz data...")
    final_quiz = structure_quiz(role, quiz_data)
    output_file = "quiz.json"
    with open(output_file, "w", encoding='utf-8') as file:
        file.write(final_quiz)

    print(f"Quiz generated successfully and saved as {output_file}!")

except FileNotFoundError as fnf_error:
    print(f"File Error: {fnf_error}")
except ValueError as val_error:
    print(f"Value Error: {val_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


from bs4 import BeautifulSoup
import requests
import os

# Function to fetch and parse HTML
def fetch_and_parse(url=None, html_file=None):
    try:
        if url:
            print(f"Fetching content from URL: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Check for errors
            html_content = response.text
        elif html_file:
            print(f"Reading HTML from file: {html_file}")
            if not os.path.exists(html_file):
                raise FileNotFoundError(f"File not found: {html_file}")
            with open(html_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
        else:
            raise ValueError("Either a valid URL or a local HTML file path must be provided.")
        return BeautifulSoup(html_content, 'html.parser')
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching URL: {e}")
    except FileNotFoundError as e:
        raise ValueError(f"Error reading file: {e}")

# Function to extract data
def extract_data(soup):
    try:
        extracted_data = []
        # Find all <h5> tags containing <a> with specific attributes
        h5_tags = soup.find_all('h5', recursive=True)
        for h5 in h5_tags:
            a_tag = h5.find('a', class_='card-title-link')  # Locate <a> inside <h5>
            if a_tag:
                title = a_tag.get_text(strip=True)
                link_id = a_tag.get('id')
                data_cy = a_tag.get('data-cy')
                extracted_data.append({
                    'title': title,
                    'id': link_id,
                    'data-cy': data_cy
                })
        if not extracted_data:
            raise ValueError("No matching <h5> or <a> tags found.")
        return extracted_data
    except Exception as e:
        raise ValueError(f"Error extracting data: {e}")

# Main function
def main():
    try:
        print("Scraping job descriptions...")
        # Prompt for URL or local file
        use_url = input("Do you want to scrape from a URL? (yes/no): ").strip().lower()
        if use_url == 'yes':
            url = input("Enter the URL: ").strip()
            html_file = None
        else:
            url = None
            html_file = input("Enter the path to your local HTML file: ").strip()

        # Fetch and parse HTML
        soup = fetch_and_parse(url=url, html_file=html_file)

        # Extract data
        data = extract_data(soup)

        # Print results
        print("\nExtracted Data:")
        for entry in data:
            print(f"Title: {entry['title']}, ID: {entry['id']}, Data-CY: {entry['data-cy']}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

# Run the script
if __name__ == '__main__':
    main()
