"""Questionnaire items that are analysed in the questionnaire study (core/survey_study.py) but are not
model inputs: perceptions, behaviours, gap checklists, artifact requirements and the items that repeat
a model question in other words (used as consistency checks).

Each item: (field name, questionnaire item, kind, [(category key, form label), ...] or None, description).
Kinds: "likert" 1-5, "categorical" one choice, "binary" yes/no, "multi" select-all (stored as
semicolon-separated keys), "text" free text. core/schema.py turns these into FieldSpecs and
core/forms_adapter.py uses the labels to read Google Forms exports.
"""
from __future__ import annotations

OTHER = ("other", "Other")

GRADUATE_ITEMS = [
    # v2.2: language of study and English proficiency (UGC/USJ 2023 census; Kelaniya 2022) - descriptive only
    ("instruction_medium", "Q7a", "categorical", [
        ("english", "English"), ("sinhala", "Sinhala"), ("tamil", "Tamil"),
        ("mixed", "Mixed (English and Sinhala / Tamil)")],
     "Main medium of instruction in the degree"),
    ("english_proficiency", "Q7b", "likert", None,
     "English proficiency (speaking and writing) at degree completion (1-5)"),
    ("degree_relevance", "Q8", "likert", None, "Perceived relevance of the degree to industry (1-5)"),
    ("cert_areas", "Q10", "multi", [
        ("cloud", "Cloud Computing (AWS / Azure / GCP)"), ("cybersecurity", "Cybersecurity"),
        ("networking", "Networking"), ("database", "Database"), ("programming", "Programming"),
        ("data_ai", "Data Science / AI"), ("agile_scrum", "Agile / Scrum"), OTHER],
     "Certification areas completed"),
    ("portfolio_update_freq", "Q15", "categorical", [
        ("never", "Never"), ("rarely", "Rarely"), ("occasionally", "Occasionally"), ("frequently", "Frequently")],
     "How often the technical portfolio is updated"),
    ("learning_platforms", "Q22", "multi", [
        ("coursera", "Coursera"), ("udemy", "Udemy"), ("linkedin_learning", "LinkedIn Learning"),
        ("google", "Google Learning"), ("microsoft_learn", "Microsoft Learn"), ("aws_training", "AWS Training"),
        ("youtube", "YouTube technical channels"), ("ai_tools", "AI tools (e.g. ChatGPT, GitHub Copilot)"), OTHER],
     "Online learning platforms used"),
    ("languages", "Q25", "multi", [
        ("java", "Java"), ("python", "Python"), ("javascript", "JavaScript"), ("c_cpp", "C / C++"),
        ("csharp", "C#"), ("php", "PHP"), ("kotlin", "Kotlin"), OTHER],
     "Programming languages used confidently (repeats Q18 Java/Python/JavaScript)"),
    ("project_areas", "Q26", "multi", [
        ("web", "Web applications"), ("mobile", "Mobile applications"), ("desktop", "Desktop applications"),
        ("data_analytics", "Data analytics"), ("machine_learning", "Machine learning"),
        ("cloud", "Cloud applications"), ("cybersecurity", "Cybersecurity projects"), ("iot", "IoT projects"),
        ("none", "None")],
     "Development areas with completed projects"),
    ("problem_confidence", "Q27", "likert", None,
     "Confidence solving an unfamiliar problem (1-5; repeats Q19 Problem Solving)"),
    ("dev_practices", "Q28", "multi", [
        ("version_control", "Version control (Git)"), ("agile_scrum", "Agile / Scrum"),
        ("code_reviews", "Code reviews"), ("testing", "Testing practices"), ("documentation", "Documentation"),
        ("cicd", "CI/CD"), ("none", "None")],
     "Industry practices used (repeats Q18 Git/Testing/DevOps and Q29)"),
    ("improvement_areas", "Q30", "multi", [
        ("programming", "Programming"), ("database", "Database"), ("cloud", "Cloud Computing"),
        ("cybersecurity", "Cybersecurity"), ("devops", "DevOps"), ("ai_ml", "AI / Machine Learning"),
        ("communication", "Communication"), ("problem_solving", "Problem Solving"),
        ("interview_skills", "Interview Skills"), ("professional_experience", "Professional Experience")],
     "Areas the graduate believes need improvement"),
    ("lacked_skills", "Q31", "multi", [
        ("programming_language", "Programming language"), ("framework", "Framework knowledge"),
        ("cloud", "Cloud technologies"), ("database", "Database skills"), ("git", "Git / GitHub"),
        ("communication", "Communication skills"), ("problem_solving", "Problem solving"),
        ("experience", "Experience"), ("certifications", "Certifications")],
     "Skills employers or interviewers said were lacking"),
    ("explain_nontechnical", "Q32", "likert", None,
     "Comfort explaining technical concepts to non-technical people (1-5; repeats Q19 Communication)"),
    ("team_confidence", "Q33", "likert", None,
     "Confidence working in a development team (1-5; repeats Q19 Teamwork)"),
    ("group_projects", "Q34", "binary", None, "Participated in group software projects"),
    ("group_project_count", "Q34a", "categorical", [("1-2", "1-2"), ("3-5", "3-5"), ("5+", "More than 5")],
     "Number of group software projects (if Q34 = yes)"),
    ("learning_strategies", "Q35", "multi", [
        ("online_courses", "Online courses"), ("documentation", "Documentation reading"),
        ("practice_projects", "Practice projects"), ("mentors", "Asking mentors"),
        ("community", "Community discussions"), ("university", "University resources"), ("ai_tools", "AI tools (e.g. ChatGPT, GitHub Copilot)")],
     "How a missing skill is usually learned"),
    ("job_platforms", "Q36", "multi", [
        ("linkedin", "LinkedIn"), ("topjobs", "TopJobs"), ("indeed", "Indeed"),
        ("company_sites", "Company career websites"), ("university_career", "University career services"),
        ("agencies", "Recruitment agencies"), OTHER],
     "Platforms used to search for IT employment"),
    ("cv_update_freq", "Q37", "categorical", [
        ("never", "Never"), ("yearly", "Once per year"), ("few_months", "Every few months"),
        ("every_application", "Before every application")],
     "How often the CV is updated"),
    ("linkedin_profile", "Q38", "binary", None, "Has a LinkedIn profile"),
    ("linkedin_completeness", "Q38a", "likert", None, "LinkedIn profile completeness (1-5, if Q38 = yes)"),
    ("interview_feedback", "Q39", "binary", None, "Received feedback from employers after interviews"),
    ("feedback_areas", "Q39a", "multi", [
        ("technical", "Technical skills"), ("communication", "Communication"), ("confidence", "Confidence"),
        ("cultural_fit", "Cultural fit"), ("experience", "Relevant experience"),
        ("certifications", "Certifications"), ("salary", "Salary expectations"), OTHER],
     "Areas mentioned in interview feedback (if Q39 = yes)"),
    ("self_employability", "Q40", "likert", None, "Self-perceived employability (1-5)"),
    ("believed_factors", "Q41", "multi", [
        ("technical_skills", "Technical skills"), ("degree_class", "Degree classification"),
        ("internship", "Internship experience"), ("projects", "Personal projects"),
        ("certifications", "Certifications"), ("communication", "Communication skills"),
        ("networking", "Networking"), ("interview_prep", "Interview preparation"),
        ("university_reputation", "University reputation"), ("industry_connections", "Industry connections")],
     "Factors believed to influence employability most (up to 5)"),
    ("university_prepared", "Q42", "likert", None, "University prepared me for industry (1-5)"),
    ("biggest_challenge", "Q43", "categorical", [
        ("technical_skills", "Lack of technical skills"), ("experience", "Lack of experience"),
        ("confidence", "Lack of confidence"), ("industry_knowledge", "Lack of industry knowledge"),
        ("job_opportunities", "Lack of job opportunities"), ("interviews", "Interview difficulties"), OTHER],
     "Biggest challenge entering the IT job market"),
    ("explanation_importance", "Q44", "likert", None, "Importance of understanding the reasons for a prediction (1-5)"),
    ("explanation_types", "Q45", "multi", [
        ("strengths", "Skills I am strong in"), ("improvements", "Skills I need to improve"),
        ("industry_comparison", "Comparison with industry requirements"),
        ("learning_paths", "Recommended learning paths"), ("similar_profiles", "Similar successful graduate profiles")],
     "Most useful explanation types"),
    ("would_use_system", "Q46", "likert", None, "Would use an AI employability system before applying (1-5)"),
    ("comments", "Q47", "text", None, "Additional comments (optional)"),
]

EMPLOYER_RANK_FACTORS = [   # Q20: rank top 3 (1 = most important)
    ("technical_depth", "Technical skill depth"), ("portfolio", "Portfolio / project quality"),
    ("interview", "Interview performance"), ("certifications", "Professional certifications"),
    ("internship", "Internship experience"), ("communication", "Communication skills"),
    ("problem_solving", "Demonstrated problem-solving ability"), ("cultural_fit", "Cultural fit"),
]

EMPLOYER_ITEMS = [
    ("respondent_role", "Q3", "categorical", [
        ("hr_manager", "HR Manager"), ("recruiter", "Recruiter"), ("tech_lead", "Technical Lead"),
        ("engineer_hiring", "Software Engineer involved in hiring"), ("eng_manager", "Engineering Manager"),
        ("project_manager", "Project Manager"), ("cto_cio", "CTO / CIO"), OTHER],
     "Respondent's role"),
    ("recruit_experience", "Q4", "categorical", [
        ("<1", "Less than 1 year"), ("1-3", "1-3 years"), ("4-6", "4-6 years"), ("7-10", "7-10 years"),
        ("10+", "More than 10 years")],
     "Years of IT recruitment experience"),
    ("roles_recruited", "Q5", "multi", [
        ("software_engineer", "Software Engineer"), ("qa", "QA Engineer"), ("data_analyst", "Data Analyst"),
        ("data_scientist", "Data Scientist"), ("ai_ml_engineer", "AI / ML Engineer"),
        ("cloud", "Cloud Engineer"), ("devops", "DevOps Engineer"), ("cybersecurity", "Cybersecurity Analyst"),
        ("business_analyst", "Business Analyst"), ("network_admin", "Network / System Administrator"),
        ("ui_ux", "UI / UX Designer"), OTHER],
     "Entry-level IT roles recruited"),
    ("annual_hires", "Q6", "categorical", [
        ("0", "None"), ("1-5", "1-5"), ("6-10", "6-10"), ("11-25", "11-25"), ("25+", "More than 25")],
     "Entry-level IT graduates recruited per year"),
    # v2.2: English proficiency (pairs with graduate Q7b) - descriptive only
    ("emp_english_importance", "Q9a", "likert", None,
     "Importance of English proficiency for entry-level IT graduates (1-5)"),
    ("future_tech", "Q10", "multi", [
        ("ai", "Artificial Intelligence"), ("ml", "Machine Learning"), ("cloud", "Cloud Computing"),
        ("cybersecurity", "Cybersecurity"), ("data_engineering", "Data Engineering"), ("devops", "DevOps"),
        ("automation", "Automation"), ("blockchain", "Blockchain"), ("iot", "IoT"),
        ("software_engineering", "Software Engineering")],
     "Technologies expected to grow in importance (3-5 years)"),
    ("skill_update_freq", "Q11", "categorical", [
        ("more_than_yearly", "More often than once a year"), ("1-2_years", "Every 1-2 years"),
        ("3-5_years", "Every 3-5 years"), ("rarely", "Rarely")],
     "How often required technical skills are updated"),
    ("tech_weaknesses", "Q12", "multi", [
        ("programming", "Programming ability"), ("se_practices", "Software engineering practices"),
        ("database", "Database knowledge"), ("cloud", "Cloud technologies"), ("testing", "Testing skills"),
        ("cybersecurity", "Cybersecurity knowledge"), ("devops", "DevOps practices"),
        ("ai_ml", "AI / ML knowledge"), ("data_analysis", "Data analysis"), OTHER],
     "Technical weaknesses observed in fresh graduates"),
    ("soft_weaknesses", "Q13", "multi", [
        ("communication", "Communication"), ("teamwork", "Teamwork"), ("confidence", "Confidence"),
        ("problem_solving", "Problem solving"), ("adaptability", "Adaptability"),
        ("professional_behaviour", "Professional behaviour"), ("time_management", "Time management"),
        ("attitude", "Attitude / lack of interest in the job")],
     "Soft-skill weaknesses observed in fresh graduates"),
    ("grad_alignment", "Q14", "likert", None, "How well graduates meet expectations (1-5)"),
    ("uni_improvements", "Q15", "multi", [
        ("industry_projects", "Industry-based projects"), ("internships", "Internship opportunities"),
        ("modern_tech", "Modern technologies"), ("soft_skills", "Soft skills training"),
        ("career_prep", "Career preparation"), ("certifications", "Professional certifications"),
        ("practical_dev", "Practical software development")],
     "Areas universities should improve"),
    ("ai_usefulness", "Q16", "likert", None, "Usefulness of an AI employability system (1-5)"),
    ("emp_explanation_importance", "Q17", "likert", None, "Importance of explanations from an AI system (1-5)"),
    ("ai_outputs", "Q18", "multi", [
        ("strengths", "Candidate skill strengths"), ("missing_skills", "Missing skills"),
        ("suitability_score", "Industry-suitability score"), ("training_areas", "Recommended training areas"),
        ("job_roles", "Suitable job roles"), ("interview_prep", "Interview preparation guidance")],
     "Useful AI-generated outputs"),
    ("hiring_challenges", "Q19", "multi", [
        ("tech_skills", "Lack of technical skills"), ("curriculum_gap", "Skill gap between curriculum and industry needs"),
        ("soft_skills", "Weak communication / soft skills"), ("experience", "Lack of practical or internship experience"),
        ("salary_expectations", "Unrealistic salary expectations"), ("attrition", "High early-stage attrition / turnover"),
        ("long_process", "Long screening or interview process"), OTHER],
     "Biggest challenges when hiring fresh graduates (up to 3)"),
    *[(f"rank_{k}", "Q20", "rank", [(k, lab)], f"Rank of '{lab}' among the top 3 (1 = most important)")
      for k, lab in EMPLOYER_RANK_FACTORS],
    ("emp_comments", "Q21", "text", None, "Additional comments (optional)"),
]


def options(items, name: str) -> list[str]:
    return [k for k, _ in next(o for n, _, _, o, _ in items if n == name)]
