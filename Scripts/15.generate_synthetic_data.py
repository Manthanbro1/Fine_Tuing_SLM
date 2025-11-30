import json
import random
import os

# ----------------------------
# Utility: random generators
# # ----------------------------
FIRST_NAMES = [
    "Riya", "Aarav", "Manthan", "Neha", "Karan", "Zoya", "Kabir", "Saanvi", "Ishaan", "Diya",
    "Arjun", "Ayesha", "Vivaan", "Zara", "Advik", "Kiara", "Dev", "Ananya", "Reyansh", "Myra",
    "Aditi", "Rohan", "Pooja", "Gaurav", "Simran", "Aryan", "Janhvi", "Dhruv", "Alia", "Viraj"
]
LAST_NAMES = [
    "Patel", "Sharma", "Gupta", "Verma", "Joshi", "Mehta", "Khan", "Bose", "Reddy", "Singh",
    "Kumar", "Iyer", "Nair", "Shah", "Choudhury", "Malik", "Bhatia", "Wadhwa", "Srinivasan", "Das",
    "Tiwari", "Yadav", "Rajput", "Sahoo", "Dutta", "Goswami", "Pillai", "Mishra", "Chopra", "Rao"
]

SKILLS_GENERAL = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "Data Analysis",
    "React", "Node.js", "Time Management", "Leadership",
    "Docker", "TensorFlow", "Pandas", "NLP",
    "Java", "C++", "JavaScript", "Go", "AWS", "Azure", "Google Cloud (GCP)", "NoSQL",
    "PostgreSQL", "Tableau", "Power BI", "Communication", "Agile", "Scrum",
    "Git", "Kubernetes", "PyTorch", "Scikit-learn", "Computer Vision", "ETL", "API Development",
    "C#", "Vue.js", "MongoDB", "Figma", "Microservices", "System Design", "Cloud Security",
    "Data Visualization", "Jupyter", "Bash", "R", "Shell Scripting"
]

COMPANIES = [
    "TechNova", "InstaData", "MegaSoft", "BlueBridge", "SkyNet", "NextGen Systems",
    "GlobalTech Solutions", "Alpha Analytics", "CloudSphere", "Pioneer Labs", "Stellar Corp", "OmniVerse Inc.",
    "DataStream", "FutureWave", "Innovatech", "Peak Systems", "Veridian Dynamics", "Quantum Leap"
]
ROLES = [
    "Software Engineer", "Data Analyst", "ML Engineer", "Backend Developer", "AI Researcher", "Data Scientist",
    "Frontend Developer", "DevOps Engineer", "Product Manager", "Business Analyst", "UX Designer", "Cloud Architect",
    "Cybersecurity Specialist", "Technical Writer", "Database Administrator", "Network Engineer", "Project Manager", "SRE"
]
""" Use below for medical dataset"""
# FIRST_NAMES = [
#     "Aarav", "Priya", "Ankit", "Sana", "Kabir", "Diya", "Rohan", "Meera", "Vikas", "Zoya",
#     "John", "Sarah", "David", "Emily", "Michael", "Olivia", "Daniel", "Sophia", "James", "Ava",
#     "Mohammed", "Fatima", "Ali", "Aisha", "Wei", "Li", "Kenji", "Yuki", "Jose", "Maria"
# ]
# LAST_NAMES = [
#     "Sharma", "Patel", "Singh", "Verma", "Gupta", "Khan", "Kumar", "Bose", "Reddy", "Mehta",
#     "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
#     "Chen", "Wang", "Kim", "Tanaka", "Silva", "Santos", "Dubois", "Schmidt", "Rossi", "Novak"
# ]
#
# SKILLS_GENERAL = [
#     "Patient Care", "Clinical Documentation", "Surgical Assistance", "Infection Control",
#     "Pharmacology", "Diagnostic Imaging", "Electronic Health Records (EHR)", "Triage",
#     "Phlebotomy", "Cardiopulmonary Resuscitation (CPR)", "Emergency Response", "Health Informatics",
#     "Biostatistics", "Medical Research", "Data Analysis (Clinical)", "Epidemiology",
#     "Medical Coding (ICD-10)", "Billing Procedures", "Regulatory Compliance (HIPAA)", "Team Leadership (Ward)"
# ]
#
# COMPANIES = [
#     "City General Hospital", "Mercy Medical Center", "Regional Health Clinic", "St. Jude's Specialty",
#     "Apex Trauma Center", "Community Care Unit", "Northwest Pediatrics", "Veterans Memorial Hospital",
#     "Global Wellness Center", "University Research Hospital",
#     "PharmaCo Research Labs", "MediTech Devices Inc.", "BioGen Pharmaceuticals",
#     "HealthData Solutions", "NextGen EHR Systems", "The Wellness Group Clinics"
# ]
# ROLES = [
#     "Physician (MD/DO)", "Registered Nurse (RN)", "Nurse Practitioner (NP)", "Physician Assistant (PA)",
#     "Medical Assistant (MA)", "Surgeon", "Pharmacist", "Physical Therapist (PT)",
#     "Radiologist", "Laboratory Technician", "Health Information Technician", "Medical Biller/Coder",
#     "Cardiologist", "Neurologist", "Oncologist", "Pediatrician",
#     "Medical Researcher", "Clinical Trial Coordinator", "Healthcare Administrator", "Biostatistician"
# ]
# ----------------------------
# Helper functions
# ----------------------------
def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def random_email(name):
    base = name.lower().replace(" ", ".")
    domains = ["@gmail.com", "@yahoo.com", "@hotmail.com", "@outlook.com"]
    return base + random.choice(domains)

def random_skills():
    num = random.randint(3, 6)
    return random.sample(SKILLS_GENERAL, num)

def pick_template(name, email, role, company, years, skills_list):
    templates = [
        # Template 1 — Direct Description
        f"{name} has worked as a {role} at {company} for {years} years. "
        f"Their skills include {skills_list}. Contact: {email}.",

        # Template 2 — Narrative Style
        f"For the past {years} years, I have been a {role} at {company}. "
        f"I'm {name}, reachable at {email}. I specialize in {skills_list}.",

        # Template 3 — Bullet Style but inline
        f"Name: {name} | Email: {email} | Role: {role} at {company} | "
        f"Experience: {years} years | Skills: {skills_list}",

        # Template 4 — Mixed Order
        f"You can contact {name} at {email}. Their expertise includes {skills_list}. "
        f"They currently work as a {role} at {company} with {years} years of experience."
    ]
    return random.choice(templates)

# ----------------------------
# Main generator
# ----------------------------
def generate_resume_sample():
    name = random_name()
    email = random_email(name)
    role = random.choice(ROLES)
    company = random.choice(COMPANIES)
    years = round(random.uniform(0.5, 12.0), 1)
    skills = random_skills()
    skills_list = ", ".join(skills)

    # Select random template
    input_text = pick_template(name, email, role, company, years, skills_list)

    # Structured JSON output
    output_json = {
        "name": name,
        "email": email,
        "skills": skills,
        "experience": [
            {
                "company": company,
                "role": role,
                "years": years
            }
        ]
    }

    return {"input": input_text, "output": output_json}

# ----------------------------
# Generate dataset
# ----------------------------
def generate_dataset(file_path, count):
    print(f"Generating {count} synthetic samples -> {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for _ in range(count):
            json.dump(generate_resume_sample(), f)
            f.write("\n")
    print("Completed.")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    generate_dataset("E:/College/2nd Year/Sem 1/EDAI/Project/Data/Resume/resume_synthetic.jsonl", 800)
