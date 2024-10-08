from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)


# Step 1: Load and Clean the Data
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully")

    relevant_data = data[
        [
            "What are your skills ? (Select multiple if necessary)",
            "If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA.               ",
        ]
    ]
    relevant_data.columns = ["Skills", "Job_Title"]
    relevant_data = relevant_data.dropna(subset=["Job_Title", "Skills"])
    relevant_data["Skills"] = relevant_data["Skills"].str.split(r"[;,]")
    relevant_data = relevant_data.explode("Skills")
    relevant_data["Skills"] = (
        relevant_data["Skills"].str.strip().str.lower()
    )  # Normalize skills to lowercase
    relevant_data = relevant_data[relevant_data["Skills"] != ""]
    relevant_data = relevant_data.dropna(subset=["Skills"])

    # Step 2: Create the Skills-to-Job Profiles Mapping
    skill_to_jobs_mapping = (
        relevant_data.groupby("Skills")["Job_Title"]
        .apply(lambda x: list(x.unique()))
        .reset_index()
    )
    skill_to_jobs_mapping.columns = ["Skill", "Job_Titles"]

    print("Mapping created successfully")
    return skill_to_jobs_mapping


skill_to_jobs_mapping = load_and_prepare_data("career_recommender.csv")

# Categorize skills
skill_categories = {
    "Programming Languages": ["java", "python", "c++", "c#", "javascript", "sql"],
    "Web Development": ["html", "css", "javascript", "react.js", "angular", "node.js"],
    "Data Science": [
        "machine learning",
        "data analysis",
        "data visualization",
        "r",
        "sql",
    ],
    "Networking & Security": ["network security", "ethical hacking", "cybersecurity"],
    "Project Management": ["project management", "agile", "scrum"],
    "Marketing & Sales": ["digital marketing", "content creation", "seo", "salesforce"],
    "Others": ["photoshop", "excel", "accounting", "business analysis"],
}

# List of industries
industries = [
    "Technology",
    "Finance",
    "Healthcare",
    "Education",
    "Manufacturing",
    "Marketing",
    "Others",
]


# Step 3: Implement the Recommendation Logic
def recommend_jobs(user_skills, skill_to_jobs_mapping, industry=None):
    recommended_jobs = []
    for skill in user_skills:
        skill = skill.strip().lower()  # Normalize input skill to lowercase
        if skill in skill_to_jobs_mapping["Skill"].values:
            jobs = skill_to_jobs_mapping[skill_to_jobs_mapping["Skill"] == skill][
                "Job_Titles"
            ].values[0]
            if industry:
                jobs = [job for job in jobs if industry.lower() in job.lower()]
            recommended_jobs.extend(jobs)
    return list(set(recommended_jobs))


@app.route("/", methods=["GET", "POST"])
def index():
    recommended_jobs = []
    if request.method == "POST":
        user_skills = request.form.getlist("skills")  # Get multiple selected skills
        print(f"User skills: {user_skills}")
        industry = request.form.get("industry")
        print(f"Industry: {industry}")
        recommended_jobs = recommend_jobs(user_skills, skill_to_jobs_mapping, industry)
    return render_template(
        "index.html",
        skill_categories=skill_categories,
        industries=industries,
        jobs=recommended_jobs,
    )


if __name__ == "__main__":
    app.run(debug=True)
